import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
# import Hexapod
import matplotlib.pyplot as plt
import numpy as np
import time

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value).double(), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std,lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.tensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state.float(), memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        # print(self.lr)
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device).double()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            advantages = (rewards - state_values.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.lr = self.lr
            # print(self.optimizer.lr)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "Hexapod-v1"
    filename = "PPO_continuous_" +env_name+ ".pth"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20      # print avg reward in the interval
    max_episodes = 8000       # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 2000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 30            # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.995                # discount factor
    
    lr_init = 0.0003                # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = 9
    save_model = 100
    #############################################
    
    # creating environment
    # env = Hexapod.Hexapod_V1(render) 
    env = gym.make('gym_hexapod:Hexapod-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # print(state_dim, action_dim)
    # env_name = "BipedalWalker-v3"

    # env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    lr = 0.0003
    ppo = PPO(state_dim, action_dim, action_std,lr, betas, gamma, K_epochs, eps_clip)
    # ppo.policy_old.load_state_dict(torch.load(filename))
    print(lr)
    
    # logging variables
    running_reward = []
    avg_length = 0
    time_step = 0
    last_best = 100
    
    # training loop
    t = time.time()
    for i_episode in range(1, max_episodes+1):

        # ppo.lr = max(lr_init*np.exp(-i_episode/700), lr_init)
        # if i_episode > 800:
            # lr_init = 0
        rewards  = []
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            action = np.clip(action,-1, 1)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                # print('yes')
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
        
            # if i_episode%20 == 0:
            #     env.render()
            if done:
                break
        # if i_episode%20 == 0:
        #         env.close()        
        
        avg_length += t
        running_reward.append(sum(rewards))
        avg_reward = np.mean(running_reward[-100:])
        
        # stop training if avg_reward > solved_reward
        if i_episode > 5 and avg_reward > last_best:
            print("########## Saving! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            last_best = avg_reward
            print(last_best)
        
        # # save every 500 episodes
        # if i_episode % save_model == 0:
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            # running_reward = int((running_reward/log_interval))
            print('Episode {}\t length:{}\treward:{:.2f}\tAvg reward:{:.2f}'.format(i_episode, avg_length,sum(rewards),avg_reward))
            avg_length = 0


    x = [i+1 for i in range(max_episodes)]
    print(time.time() - t)
    plot_learning_curve(x, running_reward)

    print('input 0 or 1')
    if int(input()) == 1:
    	print("########## Saving! ##########")
    	torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))        

            
if __name__ == '__main__':
    main()
# import torch
# import torch.nn as nn
# from torch.distributions import MultivariateNormal
# import gym
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# def plot_learning_curve(x, scores):
#     running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
#     plt.plot(x, running_avg)
#     plt.title('Running average of previous 100 scores')
#     plt.show()


# class Memory:
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.is_terminals = []

#     def clear_memory(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.is_terminals[:]


# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, action_std):
#         super(ActorCritic, self).__init__()
#         # action mean range -1 to 1
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, action_dim),
#             nn.Tanh()
#         )
#         # critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
#         self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

#     def forward(self):
#         raise NotImplementedError

#     def act(self, state, memory):
#         action_mean = self.actor(state)
#         cov_mat = torch.diag(self.action_var).to(device)

#         dist = MultivariateNormal(action_mean, cov_mat)
#         action = dist.sample()
#         action_logprob = dist.log_prob(action)

#         memory.states.append(state)
#         memory.actions.append(action)
#         memory.logprobs.append(action_logprob)

#         return action.detach()

#     def evaluate(self, state, action):
#         action_mean = self.actor(state)

#         action_var = self.action_var.expand_as(action_mean)
#         cov_mat = torch.diag_embed(action_var).to(device)

#         dist = MultivariateNormal(action_mean, cov_mat)

#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         state_value = self.critic(state)

#         return action_logprobs, torch.squeeze(state_value).double(), dist_entropy


# class PPO:
#     def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
#         self.lr = lr
#         self.betas = betas
#         self.gamma = gamma
#         self.eps_clip = eps_clip
#         self.K_epochs = K_epochs

#         self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=betas)

#         self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
#         self.policy_old.load_state_dict(self.policy.state_dict())

#         self.MseLoss = nn.MSELoss()

#     def select_action(self, state, memory):
#         state = torch.tensor(state.reshape(1, -1)).to(device)
#         return self.policy_old.act(state.float(), memory).cpu().data.numpy().flatten()

#     def update(self, memory):
#         # Monte Carlo estimate of rewards:
#         rewards = []
#         # print(self.lr)
#         discounted_reward = 0
#         for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)

#         # Normalizing the rewards:
#         rewards = torch.tensor(rewards).to(device).double()
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

#         # convert list to tensor
#         old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
#         old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
#         old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

#         # Optimize policy for K epochs:
#         for _ in range(self.K_epochs):
#             # Evaluating old actions and values :
#             logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

#             # Finding the ratio (pi_theta / pi_theta__old):
#             ratios = torch.exp(logprobs - old_logprobs.detach())
#             # Finding Surrogate Loss:
#             advantages = (rewards - state_values.detach())

#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#             loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

#             # take gradient step
#             self.optimizer.lr = self.lr
#             # print(self.optimizer.lr)
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             self.optimizer.step()

#         # Copy new weights into old policy:
#         self.policy_old.load_state_dict(self.policy.state_dict())


#  ############## Hyperparameters ##############
# def main():
#     env_name = "Hexapod-v2"
#     filename = "PPO_continuous_" + env_name + ".pth"
#     render = False
#     solved_reward = 300  # stop training if avg_reward > solved_reward

#     log_interval = 10  # print avg reward in the interval
#     max_episodes = 4000  # max training episodes
#     max_timesteps = 1500  # max timesteps in one episode

#     update_timestep = 2000  # update policy every n timesteps
#     action_std = 0.5  # constant std for action distribution (Multivariate Normal)
#     K_epochs = 30  # update policy for K epochs

#     eps_clip = 0.2  # clip parameter for PPO
#     gamma = 0.995  # discount factor
#     lr_init = 0.0003  # parameters for Adam optimizer
#     betas = (0.9, 0.999)
#     random_seed = 9
#     save_model = 100

#     #############################################

#     # creating environment
#     # env = Hexapod.Hexapod_V1(render) 
#     # state_dim = env.observation_space_shape()
#     # action_dim = env.action_space_shape()

#     env = gym.make(env_name)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     if random_seed:
#         print("Random Seed: {}".format(random_seed))
#         torch.manual_seed(random_seed)
#         np.random.seed(random_seed)

#     memory = Memory()
#     lr = 0.0003
#     ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
#     # ppo.policy_old.load_state_dict(torch.load(filename))
#     print(lr)

#     # logging variables
#     running_reward = []
#     avg_length = 0
#     time_step = 0
#     last_best = 100

#     # training loop
#     ti = time.time()
#     inp = 1
#     while inp == 1:
#         for i_episode in range(1, max_episodes + 1):
#             rewards = []
#             state = env.reset()
#             for t in range(max_timesteps):
#                 time_step += 1
#                 action = ppo.select_action(state, memory)
#                 action = np.clip(action, -1, 1)
#                 state, reward, done, _ = env.step(action)
#                 rewards.append(reward)
#                 memory.rewards.append(reward)
#                 memory.is_terminals.append(done)

#                 if time_step % update_timestep == 0:
#                     ppo.update(memory)
#                     memory.clear_memory()

#                 if done:
#                     break

#             avg_length += t
#             running_reward.append(sum(rewards))
#             avg_reward = np.mean(running_reward[-100:])

#             # if i_episode > 5 and avg_reward > last_best:
#             #     print("Saving!")
#             #     torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
#             #     last_best = avg_reward
#             #     print(last_best)

#             if i_episode % log_interval == 0:

#                 avg_length = int(avg_length / log_interval)
#                 print('Episode {}\t length:{}\treward:{:.2f}\tAvg reward:{:.2f}'.format(i_episode, avg_length, np.mean(running_reward[-20:]),avg_reward))
#                 avg_length = 0
#                 if i_episode > 5 and avg_reward > last_best:
#                     print("Saving!")
#                     torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
#                     last_best = avg_reward
#                     print(last_best)


#         print((time.time() - ti) / 3600)
#         plt.plot(running_reward)
#         plt.show()

#         print('input 0 or 1')
#         try:
#             inp = int(input())
#         except:
#             inp = int(input())

#     print('input 0 or 1')
#     if int(input()) == 1:
#         print("########## Saving! ##########")
#         torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))        

# if __name__ == '__main__':
#     main()