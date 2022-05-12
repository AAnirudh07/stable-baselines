import gym
from stable_baselines3 import PPO
import torch as th
env = gym.make("Acrobot-v1")

print("Observation Space: ", env.observation_space.shape)
print("Observation Space Low: ", env.observation_space.low)
print("Observation Space High: ", env.observation_space.high)
print("Observation Space Sample: ", env.observation_space.sample())

print("Action Space: ", env.action_space.n)
print("Action Space Sample: ", env.action_space.sample())

my_policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[dict(pi=[128, 128, 64, 64, 32], vf=[128, 128, 64, 64, 32])])
model = PPO("MlpPolicy", env, policy_kwargs = my_policy_kwargs, verbose=1)
model.learn(total_timesteps=100000)

for episode in range(3):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(state)
        state, reward, done, _  = env.step(action)
        score += reward
        env.render()
    print(f"episode: {episode}   score: {score}")
env.close()