import gym
from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy

env = gym.make('gym_hexapod:Hexapod-v3')

model = SAC(LnMlpPolicy, env, verbose=1, seed=1000, n_cpu_tf_sess=1)
model.learn(total_timesteps=1000000)
model.save("hexapod_sac")

'''
obs = env.reset()
done = False
while not done:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)    
'''

env.close()
