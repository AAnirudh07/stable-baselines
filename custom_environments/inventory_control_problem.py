'''
Problem statement:
Consider the problem of day-to-day control of an inventory of a fixed maximum size in the face of
uncertain demand: Every evening, the decision maker must decide about the quantity to be ordered 
for the next day. In the morning, the ordered quantity arrives with which the inventory is filled up. 
During the day, some stochastic demand is realized, where the demands are independent with a common 
fixed distribution. The goal of the inventory manager is to manage the inventory so as to maximize 
the present monetary value of the expected total future income.

Goal: To maximize the value of the expected total future income

The environment is discrete. The author uses the poisson distribution to model product demand.

Environment description:
    n - countable number of actions & states (countable MDP)
    t - time step (day number)

    cost of purchasing x items: K*I(x>0) + c*x where
        K - fixed cost of ordering a non-zero number of items
        c - unit price of an item
    
    cost of holding inventory of size y is h*y where
        h - proportionality factor

    incentive to the inventory manager on selling z units of product is p*z where
        p - proportionality factor

    If a represents the number of units of product ordered in the evening:
        x_new = max( min(x+a,max_inventory_size) - next_day_demand), 0 )
    
    reward function:
        1. Cost of holding excess inventory (-) : h * x + p
        2. Cost of replenising stock in the evening (-) : k * (a > 0) + c * max(min(x + a, max_inventory_size) - x, 0)
        3. Cost of selling product to consumer (+) : p * max(min(x + a, max_inventory_size) - y, 0)

        reward = -k * (a > 0) - c * max(min(x + a, max_inventory_size) - x, 0) - h * x + p * max(min(x + a, max_inventory_size) - y, 0)

    Constraints:
        0 <= a,x,y,z <= max_inventory_size
        p > h

    Termination:
        The environment terminates once the number of days reaches 100.
'''

import os
import gym
from gym import spaces
from gym import utils
import numpy as np

LAMBDA = 7
MAX_INVENTORY_SIZE = 200

class inventoryControlEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    #Initialize the environment parameters 
    def __init__(self):
        super(inventoryControlEnvironment, self).__init__()
        self.n = MAX_INVENTORY_SIZE
        self.K = 4
        self.c = 2
        self.h = 3
        self.p = 4
        self.l = LAMBDA
        self.done = False
        self.t = 0 

        #verify the constraints
        assert self.n > 0
        assert self.K > 0
        assert self.c > 0
        assert self.h > 0
        assert self.p > 0
        assert self.p > self.h

        self.action_space = spaces.Discrete(self.n+1)
        self.observation_space = spaces.Discrete(self.n+1)
        self.state = self.n  #initial state of the environment (max. inventory)

        #reset the state
        self.reset()

    def reset(self):
        self.state = MAX_INVENTORY_SIZE
        self.done = False
        return self.state


    def step(self, action): #action corresponds to the number of units of product purchased by the manager in the evening

        assert self.action_space.contains(action)
        current_state = self.state

        #simulate the demand for product
        demand = np.random.poisson(lam=self.l)

        new_state = max(min(current_state + action, MAX_INVENTORY_SIZE) - demand, 0)
        self.state = new_state

        self.reward =  -self.K * (action > 0) - self.c * max(min(current_state + action, MAX_INVENTORY_SIZE) - current_state, 0) - self.h * current_state + self.p * max(min(current_state + action, MAX_INVENTORY_SIZE) - new_state, 0)
        self.t += 1
        
        #insert done condition here
        if self.t == 100:
            self.done = True

        info = {}
        return self.state, self.reward, self.done, info

    def close(self):
        #no windows to close
        os.system('cls') #windows
        print("Environment closed!")
    
    #print the current inventory and reward 
    def render(self):
        print(f"Current Inventory: {self.state} | Current reward: {self.reward}")
