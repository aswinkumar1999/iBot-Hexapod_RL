import gym
# import gym_hexapod


env = gym.make('gym_hexapod:Hexapod-v3')
env.reset()
for _ in range(1000):
    env.render()
    (env.step(env.action_space.sample()) )# take a random action
env.close()