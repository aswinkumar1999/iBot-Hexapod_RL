# Hexapod-V3

To make the Hexapod env:
1) Clone the repo.
2) cd Hexapod-V3
3) Use ```gym.make('gym_hexapod:Hexapod-v3')``` to make an env.
4) Try running ```hexapod_check.py```
5) Try running ```sac_run.py``` for a trained SAC hexapod agent. (Make sure to get stable-baselines and tensorflow==1.14 first!)
6) Try running ```PPO_run.py``` for a trained PPO hexapod agent. (Pytorch)

### Debugging : 

***GLEW initalization error: Missing GL version***

Run the following : 

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
