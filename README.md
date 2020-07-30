# iBot-Hexapod_RL

# Hexapod V3

- Made it into a GYM Environment. 

# Hexapod V2

- Added a Larger Plane
- Correct the Motor Torque Outputs
- Adjusted the Damping and Stiffness factor of the Bot
- Added Transformation matrix for Easier SIM2REAL deployment.
- General Stability of the bot improved and resembles much more closer to the real bot. 

# Hexapod V1 

- Modelled using STL files.
- Hexapod Model with Basic Functionality and Python Interface

# Installation Instructions 

1. Get MuJoCo from the MuJoCo Website

2. Install MuJoCo-Py ( Using pip or [from source](https://aswinkumar1999.github.io/robotics/mujoco/mujoco-py/openai/2020/05/15/mujoco-part-0/#/))

3. Clone this repository 

```bash
cd ~/.mujoco/mujoco200/
git clone https://github.com/aswinkumar1999/iBot-Hexapod_RL.git
mv iBot-Hexapod_RL/* .
```

To test if everything works 

```bash
cd ~/.mujoco/mujoco200/Hexapod_V1=3
python3 hexapod_triple_gait.py
```

It should open up a window, with the Hexapod walking using Triple Gait... Check out the code for all the class implementations.. 
