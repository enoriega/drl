import gym

env = gym.make('Pong-v0')
env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print reward, done
    if done: break
