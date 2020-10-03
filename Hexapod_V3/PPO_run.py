import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach()
  

class PPO:
    def __init__(self, state_dim, action_dim, action_std):
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)

    
    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).to(device)
        return self.policy.act(state.float()).cpu().data.numpy().flatten()
    

def main():
    ############## Hyperparameters ##############

    env_name = "Hexapod-v3"
    env = gym.make('gym_hexapod:Hexapod-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    
    n_episodes = 5     # num of episodes to run
    max_timesteps = 15000    # max timesteps in one episode
    render = True           # render the environment
    
    # filename and directory to load model from
    filename = "PPO_continuous_" +env_name+ ".pth"

    action_std = 0.5 

    #############################################
    total_reward = []
    ppo = PPO(state_dim, action_dim, action_std)
    ppo.policy.load_state_dict(torch.load(filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state)
            action = np.clip(action,-1, 1)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            env.render()

        env.close()    
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward.append(ep_reward)
        ep_reward = 0
    print(np.mean(total_reward))  

    
if __name__ == '__main__':
    main()
