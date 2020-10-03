import gym
from PPO_continuous import PPO, Memory
from PIL import Image
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
# import Hexapod

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    ############## Hyperparameters ##############

    env_name = "Hexapod-v1"
    # env = gym.make(env_name)
    env = gym.make('gym_hexapod:Hexapod-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    
    n_episodes = 11     # num of episodes to run
    max_timesteps = 15000    # max timesteps in one episode
    render = True           # render the environment
    save_gif = False     # png images are saved in gif folder
    
    # filename and directory to load model from
    filename = "PPO_continuous_" +env_name+ ".pth"
    directory = "./preTrained/"
    # filename = directory + filename

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 3           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    li = []
    memory = Memory()
    total_reward = []
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            action = np.clip(action,-1, 1)
            li.append(action)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            env.render()
            if save_gif:
                 img = env.run_sim()
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break

        env.close()    
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward.append(ep_reward)
        ep_reward = 0
    print(np.mean(total_reward))  

    # with open('action.txt' , 'wb') as f:
    #     pickle.dump(li, f)  
    # with open("action.txt", "rb") as fp:
    #     b = pickle.load(fp)
    #     print(b)
    
if __name__ == '__main__':
    main()