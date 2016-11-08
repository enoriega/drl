import gym

env = gym.make('Pong-v0')
env.reset()
env.render()
while True:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print reward, done
    env.render()
    if done: break
