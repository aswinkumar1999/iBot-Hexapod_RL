import mujoco_py
import numpy as np
import os
import gym
from gym import spaces
from gym.envs.mujoco import mujoco_env
from gym_hexapod.envs import rotations
from gym import utils

print(os.path.dirname(__file__))


class HexapodEnv(mujoco_env.MujocoEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'assets','hexapod.xml'), 5)


    def get_orientation(self):
        return self.data.qpos[3:7]

    def get_joint_angles(self):
        return self.data.qpos[-18:]

    def get_touch_data(self):
        return self.data.sensordata


    def done(self):
        if self.data.qpos[2] > 0.05:
            return True
        if abs(self.data.qpos[1] - self.init_qpos[1]) > 0.5:
            return True 
        else:
            return False

    def observation(self):
        # obs = list(self.get_position()) + list(self.get_orientation_euler()) + list(self.get_joint_angles())
        obs = list(self.get_orientation_euler()) + list(self.get_joint_angles())
        return obs

    def step(self,action):

        self.x_last = self.sim.data.qpos[0]
        self.do_simulation(action, 200)
        #self.reward_val = self.sim.data.qpos[0] - self.x_last - 0.001*np.sqrt((((self.sim.data.qpos[3:7] - self.orient_last))** 2).mean())
        self.reward_val = self.sim.data.qpos[0].copy() - self.x_last
        return  np.array(self.observation()),100*(np.array(self.reward_val)),self.done(), {3:3}


    def get_orientation_euler(self):
        return rotations.mat2euler(rotations.quat2mat(self.sim.data.qpos[3:7]))

    def reset_model(self):
        return np.array(self.observation())
