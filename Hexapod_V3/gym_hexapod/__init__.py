from gym.envs.registration import registry, register, make, spec


#  Mujoco
# ----------------------------------------
# The Hexapod env has been registered below. 
# The max_episode_steps can be changed according to the algorithm. 
# ----------------------------------------

register(
    id='Hexapod-v3',
    entry_point='gym_hexapod.envs:HexapodEnv',
    max_episode_steps=1000,
)

