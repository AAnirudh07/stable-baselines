import gym

env = gym.make("SpaceInvaders-v0", render_mode="human")
print("Observation Space: ", env.observation_space.shape)
print("Observation Space Low: ", env.observation_space.low)
print("Observation Space High: ", env.observation_space.high)
print("Observation Space Sample: ", env.observation_space.sample())

print("Action Space: ", env.action_space.n)
print("Action Space Sample: ", env.action_space.sample())
env.unwrapped.get_action_meanings()

for episode in range(3):
    state = env.reset()
    done = False
    while not done:
        state, reward, done, _ = env.step(env.action_space.sample())
env.close()





