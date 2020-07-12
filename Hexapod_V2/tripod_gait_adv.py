#Import required libraries
import Hexapod
import math
from numpy import random

# Hardcoded values for Tripod Gait
t1=[0.3,-0.3,0.3,0.3,-0.3,0.3,0.1,0,0.1,0,-0.1,0,0,0,0,0,0,0]
t2=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0.1,0,0.1,0,-0.1,0,0,0,0,0,0,0]
t3=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0,0.1,0,-0.1,0,-0.1,0,0,0,0,0,0]
t4=[0.3,-0.3,0.3,0.3,-0.3,0.3,0,0.1,0,-0.1,0,-0.1,0,0,0,0,0,0]
# steps=[t1,t2,t3,t4]

# x ~1 , y ~ -1 Z ~ -pi/2

adv = [0,0,0,0,0,0,0.185,0.185,0.185,-0.05,0,0,0,0,0,0,0,0]

steps=[]
steps.append([a + b for a, b in zip(adv, t1)])
steps.append([a + b for a, b in zip(adv, t2)])
steps.append([a + b for a, b in zip(adv, t3)])
steps.append([a + b for a, b in zip(adv, t4)])

# Create an Object of Class Hexapod_V1
# Feel free to name the bot anything you like :P
nimbus = Hexapod.Hexapod_V1(render=False) # Defualt is FALSE

# import numpy as np
# god = np.linspace(0,0.2,11).tolist()
# # print(god)
tot =0



# for a in god:
#     for b in god:
#         for c in god:
#             for d in god:
#                 for e in god:
#                     for f in god:
#                         adv.append([0,0,0,0,0,0,a,b,c,-1*d,-1*e,-1*f,0,0,0,0,0,0])
#                         tot = tot +1
#                         print(tot)
#                         nimbus.reset()
#                         for i in range(1000):
#                             x,y,z  = nimbus.get_position()
#                             X, Y ,Z = nimbus.get_orientation_euler()
#                             # Set the Manual Tripod Gait angles
#                             nimbus.set_joint_angles(steps[i%4])
#                             # Simulate by number of 0.002s * timestep and render it on Screen.
#                             nimbus.run_sim(timestep=100)
#                             if(i > 5):
#                             # Check and break
#                                 y_mid = -1*math.sqrt(1 - x*x)
#                                 z_mid = math.atan(-x/y)-1.57
#                                 if ( y_mid-0.1 <= y <= y_mid+0.1 ) and ( z_mid-0.1 <= Z <= z_mid+0.1 ):
#                                     lis = [a,b,c,d,e,f]
#                                     print(lis)
#                                 else :
#                                     break


found = False

while (not found):
    adv = [0,0,0,0,0,0]+(random.uniform(low=-0.2,high=0.2,size=(6))).tolist()+[0,0,0,0,0,0]
    tot = tot +1
    print(tot)
    nimbus.reset()
    for i in range(1000):
        x,y,z  = nimbus.get_position()
        X, Y ,Z = nimbus.get_orientation_euler()
        # Set the Manual Tripod Gait angles
        nimbus.set_joint_angles(steps[i%4])
        # Simulate by number of 0.002s * timestep and render it on Screen.
        nimbus.run_sim(timestep=100)
        if(i > 25):
        # Check and break
            y_mid = -1*math.sqrt(1 - x*x)
            z_mid = math.atan(-x/y)-1.57
            if ( y_mid-0.1 <= y <= y_mid+0.1 ) and ( z_mid-0.1 <= Z <= z_mid+0.1 ):
                lis = [a,b,c,d,e,f]
                print(lis)
                found = True
            else :
                break
