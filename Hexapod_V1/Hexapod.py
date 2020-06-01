# Date : June 1 2020
# Author : Aswinkumar
# Description : Contains the Class file for Hexapod_V1 MuJoCo Sim


########## DO NOT ALTER THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING ###########

#Import required libraries
import mujoco_py
import rotations
import os


class Hexapod_V1:

    def __init__(self,render=False):
        # Get the mujoco-py path and Load the XML file
        self.mj_path, _ = mujoco_py.utils.discover_mujoco()
        self.xml_path = os.path.join(self.mj_path,'Hexapod_V1','assets','robot_v1.xml')
        self.model = mujoco_py.load_model_from_path(self.xml_path)
        self.render = render
        # Setup the simulator and viewer
        self.sim = mujoco_py.MjSim(self.model,nsubsteps=1)
        if(self.render):
            self.viewer = mujoco_py.MjViewer(self.sim)

    def get_position(self):
        return self.sim.data.qpos[:3]

    def get_orientation(self):
        return self.sim.data.qpos[3:7]

    def get_joint_angles(self):
        return self.sim.data.qpos[-18:]

    def get_touch_data(self):
        return self.sim.data.sensordata

    def set_joint_angles(self,action):
        self.sim.data.ctrl[:18] = action

    def get_orientation_euler(self):
        return rotations.mat2euler(rotations.quat2mat(self.sim.data.qpos[3:7]))

    def run_sim(self,timestep=1):
        for i in range(100):
            self.sim.step()
            if(self.render):
                self.viewer.render()
