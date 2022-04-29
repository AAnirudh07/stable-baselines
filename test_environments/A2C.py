import gym
import random
from stable_baselines3 import A2C

EPISODES = 10

env = gym.make("LunarLander-v2")
env.reset()
'''print("sample action: ",env.action_space.sample())
print("observation space: ",env.observation_space.shape)
print("sample observation: ",env.observation_space.sample())'''

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=40000)


for episode in range(EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action, _states = model.predict(state)
        #print(_states)
        state, reward, done, _ = env.step(action)
env.close()
