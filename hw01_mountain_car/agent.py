import os

import numpy as np

import torch

from .train import QFoo, MODEL_PATH


class Agent:
    def __init__(self):
        self.q_foo = QFoo(state_dim=2, action_dim=3)
        self.q_foo.load(os.path.join(__file__[:-8], MODEL_PATH))
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            action = self.q_foo(state).argmax(dim=1)
            return action.numpy()[0]

    def reset(self):
        pass


if __name__ == '__main__':
	agent = Agent()
	action = agent.act([0, 0])
	print(action)