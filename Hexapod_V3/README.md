# Hexapod-V3

To make the Hexapod env:
1) Clone the repo.
2) cd Hexapod-V3
3) Try running the hexapod_check.py file. 
4) Use  "gym.make('gym_hexapod:Hexapod-v3')" to make an env.

### Debugging : 

***GLEW initalization error: Missing GL version***

Run the following : 

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
