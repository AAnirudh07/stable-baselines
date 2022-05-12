import gym
from stable_baselines3 import A2C

EPISODES = 10

env = gym.make("CartPole-v1") 
state = env.reset()

#[position of cart, velocity of cart, angle of pole, rotation rate of pole]
print("Observation Space: ", env.observation_space.shape)
print("Observation Space Low: ", env.observation_space.low)
print("Observation Space High: ", env.observation_space.high)
print("Observation Space Sample: ", env.observation_space.sample())

#left or right by constant amount (0, 1)
print("Action Space: ", env.action_space.n)
print("Action Space Sample: ", env.action_space.sample())

model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50000)

for episode in range(EPISODES):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        score += reward
        env.render()
    print(f"Episode: {episode}    Score: {score}")
env.close()



