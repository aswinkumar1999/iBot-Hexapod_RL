import numpy as np
import socket
import time
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import time

device = "cpu"

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
        # action_logprob = dist.log_prob(action)
        
        return action.detach()
  

class PPO:
    def __init__(self, state_dim, action_dim, action_std):
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)

    
    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).to(device)
        return self.policy.act(state.float()).cpu().data.numpy().flatten()
    

class Sim2Real:

    def __init__(self):

        env_name = "Hexapod-v3"
        env = gym.make('gym_hexapod:Hexapod-v3')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # filename and directory to load model from
        filename = "PPO_continuous_" +env_name+ ".pth"
        action_std = 0.5 
        self.ppo = PPO(state_dim, action_dim, action_std)
        self.ppo.policy.load_state_dict(torch.load(filename))
        self.state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        # Initialise Transformation Matrix
        trans_mat = np.zeros((18,18))
        trans_mat_const = 600

        # Fill up Transformation Matrix for Sim to Real

        trans_mat[15,0]  = -1
        trans_mat[3,1]   = -1
        trans_mat[9,2]   = 1
        trans_mat[16,3]  = -1
        trans_mat[10,4]  = 1
        trans_mat[4,5]   = -1
        trans_mat[5,6]   = -1
        trans_mat[17,7]  = -1
        trans_mat[11,8]  = 1
        trans_mat[14,9]  = -1
        trans_mat[2,10]  = -1
        trans_mat[8,11]  = 1
        trans_mat[13,12] = -1
        trans_mat[1,13]  = -1
        trans_mat[7,14]  = 1
        trans_mat[0,15]  = -1
        trans_mat[6,16]  = 1
        trans_mat[12,17] = -1
        self.trans_mat = trans_mat * trans_mat_const
        
        self.m_id = [1,2,3,7,8,9,13,14,15,18,19,20,24,25,26,30,31,32]
        self.m_val= [1433, 1530, 1400, 1380, 1510, 1480, 1580, 1365, 1355, 1339, 1400, 1730, 1811, 1500, 1710, 1450, 1592, 2023]

    def reset(self):

        MESSAGE = ''
        for i in range(len(self.m_id)):
            MESSAGE=MESSAGE+'#'+str(self.m_id[i])+'P'+str(self.m_val[i])

        # print(MESSAGE)
        MESSAGE = MESSAGE + 'T200\r\n'
        return MESSAGE
        
    def step(self, orientation):

        full_state = np.array(list(self.state) + list(orientation))

        action = self.ppo.select_action(full_state)
        self.state = np.clip(action,-1, 1)
        action = list(np.array(self.state).dot(self.trans_mat))
        MESSAGE = ''
        for i in range(len(self.m_id)):
            MESSAGE=MESSAGE+'#'+str(self.m_id[i])+'P'+str(self.m_val[i]+int(action[i]))

        # print(MESSAGE)
        MESSAGE = MESSAGE + 'T200\r\n'
        return MESSAGE



if __name__ == "__main__":

    orient = [ 0.43731648,  0.19700632,  0.57860664]
    TCP_IP = '192.168.130.59'
    TCP_PORT = 80
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    print("Connected to Bot")

    bot = Sim2Real()
    message = bot.reset()
    s.send(message.encode())
    orient1 = repr(s.recv(1024))

    print('Received',orient1)
    print("Bot Setup Complete")
    input("Enter Key to Continue")

    max_steps = 10000000
    for i in range(max_steps):

        message = bot.step(orient)
        s.send(message.encode())
        orient1 = repr(s.recv(1024))
        print('Received',orient1)



