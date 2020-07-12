#Import required libraries
import Hexapod

# Hardcoded values for Tripod Gait
t1=[0.3,-0.3,0.3,0.3,-0.3,0.3,0.1,0,0.1,0,-0.1,0,0,0,0,0,0,0]
t2=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0.1,0,0.1,0,-0.1,0,0,0,0,0,0,0]
t3=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0,0.1,0,-0.1,0,-0.1,0,0,0,0,0,0]
t4=[0.3,-0.3,0.3,0.3,-0.3,0.3,0,0.1,0,-0.1,0,-0.1,0,0,0,0,0,0]
steps=[t1,t2,t3,t4]

# Create an Object of Class Hexapod_V1
# Feel free to name the bot anything you like :P
nimbus = Hexapod.Hexapod_V1(render=True) # Default is FALSE

#Resets the Hexapod and gets the default value.
observation = nimbus.reset()

# Observation = Position [x,y,z] + Euler Orientation [X,Y,Z] + Normal Force from Ground [ 6 ] + Joint Angles [ 18 ]    -> (30,) Numpy Array
# Reward = x_after_action - x_before_action
# Done = False

#
# action should either numpy array of (18,) or take a list of size 18 each value capped between [-1 1] translating to [-60 60] degrees
#

for i in range(100):
    observation, reward, done = nimbus.action(steps[i%4])
    print(observation)
    print(reward)
    print(done)

# Delete the Object
del nimbus
