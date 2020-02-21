import os

import numpy as np

import torch

from gym import make

try:
    from .train import ENV_NAME, EVAL_EPISODES_CNT, MODEL_PATH, Actor
except:
    from train import ENV_NAME, EVAL_EPISODES_CNT, MODEL_PATH, Actor


class Agent:
    def __init__(self):
        self.actor = Actor(state_dim=26, action_dim=6, max_action=1)
        self.actor.load(os.path.join(__file__[:-8], MODEL_PATH))
        
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor([state])
            action = self.actor(state)
            return action.numpy()[0]

    def reset(self):
        pass


if __name__ == '__main__':
    env = make(ENV_NAME)

    agent = Agent()

    #for _ in range(EVAL_EPISODES_CNT):
    #    state = env.reset()
    #    done = False
    #    while not done:
    #        action = aql.act(state)
    #        state, reward, done, _ = env.step(action)
    #        total_reward += reward

    #total_reward /= EVAL_EPISODES_CNT

    #print(f'reward = {total_reward}')

    total_reward = 0
    env.render()
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f'reward = {total_reward}')

    env.close()