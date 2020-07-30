import gym
# import gym_hexapod
from stable_baselines import SAC


env = gym.make('gym_hexapod:Hexapod-v3')

model = SAC.load("hexapod_sac")
obs = env.reset()
tot = 0
done = False
while not done:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    tot += rewards
    print(tot)

env.close()