import os
import gym
import random
from stable_baselines3 import A2C, PPO

model_dir = "models/PPO"    
log_dir = "logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

EPISODES = 10

env = gym.make("LunarLander-v2")
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000

for i in range(1,30): 
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{TIMESTEPS*i}")



'''for episode in range(EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action, _states = model.predict(state)
        #print(_states)
        state, reward, done, _ = env.step(action)'''
env.close()
