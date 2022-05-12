import gym
from stable_baselines3 import DQN
import tensorflow as tf

env = gym.make("Breakout-v0", render_mode="human")

print("Observation Space: ", env.observation_space.shape)
print("Observation Space Low: ", env.observation_space.low)
print("Observation Space High: ", env.observation_space.high)
print("Observation Space Sample: ", env.observation_space.sample())

print("Action Space: ", env.action_space.n)
print("Action Space Sample: ", env.action_space.sample())
print(env.unwrapped.get_action_meanings())


my_policy_kwargs = dict(net_arch=[32, 32])
model = DQN("CnnPolicy", env=env, policy_kwargs=my_policy_kwargs, verbose=1)
model.learn(total_timesteps=10000)

for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        state, reward, done, _ = env.step(env.action_space.sample())
env.close()