import os

import numpy as np

import torch

from gym import make

try:
    from .train import ENV_NAME, STATE_DIM, ACTION_DIM, EVAL_EPISODES_CNT, MODEL_PATH, QFoo
except:
    from train import ENV_NAME, STATE_DIM, ACTION_DIM, EVAL_EPISODES_CNT, MODEL_PATH, QFoo


class Agent:
    def __init__(self):
        self.q_foo = QFoo(state_dim=8, action_dim=4)
        self.q_foo.load(os.path.join(__file__[:-8], MODEL_PATH))
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            action = self.q_foo(state).argmax(dim=1)
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

    print(f'reward = {total_reward}')

    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f'reward = {total_reward}')