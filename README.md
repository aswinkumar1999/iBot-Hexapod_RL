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

- Modeled using STL files.
- Hexapod Model with Basic Functionality and Python Interface

# Installation Instructions 

## Prerequisites for Mujoco

- Install following packages and edit bashrc
```
sudo apt-get install -y libglew-dev libosmesa6-dev patchelf
conda install -c anaconda patchelf
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
source ~/.bashrc
```

### Install MuJoCo

1. Obtain a 30-day free trial on the [MuJoCo website](https://www.roboti.us/license.html)
   or free license if you are a student.
   The license key will arrive in an email with your username and password.
2. Download the MuJoCo version 2.0 binaries for
   [Linux](https://www.roboti.us/download/mujoco200_linux.zip) or
   [OSX](https://www.roboti.us/download/mujoco200_macos.zip).
3. Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`,
   and place your license key (the `mjkey.txt` file from your email)
   at `~/.mujoco/mjkey.txt`.

<!-- 4. Add following line to .bashrc:
  `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco200/bin` -->


4. Install mujoco-py `pip install -U 'mujoco-py<2.1,>=2.0'`

5. Clone this repository 

```bash
cd ~/.mujoco/mujoco200/
git clone https://github.com/aswinkumar1999/iBot-Hexapod_RL.git
mv iBot-Hexapod_RL/* .
```

To test if everything works 

```bash
cd ~/.mujoco/mujoco200/Hexapod_V3
python hexapod_triple_gait.py
```

It should open up a window, with the Hexapod walking using Triple Gait... Check out the code for all the class implementations.. 
