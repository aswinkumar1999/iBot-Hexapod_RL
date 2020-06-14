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
nimbus = Hexapod.Hexapod_V1(render=True) # Defualt is FALSE

for i in range(100):
    # Print (x,y,z) positions of the system.
    print("(x,y,z) of bot :",end="\t")
    print(nimbus.get_position())
    # Print Quaternions of the System
    print("(q1,q2,q3,q4) of bot :",end="\t")
    print(nimbus.get_orientation())
    # Print Euler Orientations of the System
    print("(X,Y,Z ) of bot :",end="\t")
    print(nimbus.get_orientation_euler())
    # Print Touch Sensor Data
    print("Sensor Data :",end="")
    print(nimbus.get_touch_data())
    # Get the joint angles
    print("Joint Angles :",end="")
    print(nimbus.get_joint_angles())
    # Set the Manual Tripod Gait angles
    nimbus.set_joint_angles(steps[i%4])
    # Simulate by number of 0.002s * timestep and render it on Screen.
    nimbus.run_sim(timestep=100)
