from stable_baselines.common.env_checker import check_env
import gym


env = gym.make('gym_hexapod:Hexapod-v3')


check_env(env)
