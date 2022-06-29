import gym

env = gym.make("Taxi-v3")
observation = env.reset()
print('observation_space',env.observation_space)
print('action space', env.action_space)

for i in range(1000):

    env.render()
    # your agent here (this takes random actions)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(i, observation, reward, done, info)
    if done:
        print(done)
        env.render()
        break
