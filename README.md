# iBot-Hexapod_RL

# Installation Instrcutions 

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
cd ~/.mujoco/mujoco200/Hexapod_V1
python3 triple_gait.py
# or 
python3 GYM_Style.py
```

It should open up a window, with the Hexapod walking using Triple Gait... Check out the code for all the class implementations.. 
