import gym
# import gym_hexapod
t1=[0.3,-0.3,0.3,0.3,-0.3,0.3,0.1,0,0.1,0,-0.1,0,0,0,0,0,0,0]
t2=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0.1,0,0.1,0,-0.1,0,0,0,0,0,0,0]
t3=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0,0.1,0,-0.1,0,-0.1,0,0,0,0,0,0]
t4=[0.3,-0.3,0.3,0.3,-0.3,0.3,0,0.1,0,-0.1,0,-0.1,0,0,0,0,0,0]
steps=[t1,t2,t3,t4]

env = gym.make('gym_hexapod:Hexapod-v3')
env.reset()
for _ in range(1000):
    env.render()
    env.step(steps[_%4]) # take a random action
env.close()
