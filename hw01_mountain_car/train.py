import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from gym import make


GAMMA = 0.96
EPSILON = 0.1
BUFFER_CAPACITY = 10000

BATCH_SIZE = 128
LR = 0.00005

TARGET_UPDATE = 1000
STEPS = 100 * TARGET_UPDATE + 1
EVAL_EPISODS = 20

MODEL_PATH = 'model.pth'


class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.array([])
        self.position = 0

    def add(self, element):
        if len(self.buffer) < self.capacity:
            self.buffer = np.concatenate((self.buffer, np.array([None])))

        self.buffer[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indexes = list(range(len(self.buffer)))
        indexes = np.random.choice(indexes, batch_size, replace=False)
        return list(self.buffer[indexes])

    def __len__(self):
        return len(self.buffer)


class QFoo(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=512, hidden_layers=1):
        super(QFoo, self).__init__()

        assert hidden_layers > 0

        layers = [nn.Linear(state_dim, hidden_size),
                  nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size),
                       nn.ReLU()]
        layers += [nn.Linear(hidden_size, action_dim)]
        self.q_foo = nn.Sequential(*layers)

    def forward(self, state):
        return self.q_foo(state)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class AQL:
    def __init__(self, state_dim, action_dim):
        self.q_foo = QFoo(state_dim, action_dim)
        self.q_foo_target = copy.deepcopy(self.q_foo)

        self.criterion = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.q_foo.parameters()),
            lr=LR
        )

        self.q_foo.eval()
        self.q_foo_target.eval()

    def update(self, batch):
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        self.q_foo.train()

        q_target = torch.zeros(BATCH_SIZE).float()
        with torch.no_grad():
            q_target[done] = self.q_foo_target(next_state).max(dim=1)[0][done]
        q_target = reward + q_target * GAMMA

        q = self.q_foo(state).gather(1, action[:, None]).squeeze()

        loss = self.criterion(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.q_foo.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            action = self.q_foo(state).argmax(dim=1)
            return action.numpy()[0]

    def save(self, path):
        self.q_foo_target = copy.deepcopy(self.q_foo)
        self.q_foo.save(path)


if __name__ == '__main__':
    env = make('MountainCar-v0')

    aql = AQL(state_dim=2, action_dim=3)

    buffer = Buffer(BUFFER_CAPACITY)

    state = env.reset()
    for step in range(STEPS):
        if np.random.rand() < EPSILON:
            action = np.random.randint(3)
        else:
            action = aql.act(state)

        new_state, reward, done, _ = env.step(action)
        modified_reward = reward + abs(new_state[1])

        buffer.add((state, action, modified_reward, new_state, done))

        if done:
            state = env.reset()
            done = False
        else:
            state = new_state

        if step > BATCH_SIZE:
            aql.update(buffer.sample(BATCH_SIZE))

        if step % TARGET_UPDATE == 0:
            aql.save(MODEL_PATH)

            total_reward = 0
            for _ in range(EVAL_EPISODS):
                state = env.reset()
                done = False
                while not done:
                    action = aql.act(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward

            total_reward /= EVAL_EPISODS

            print(f'step={step}   reward={total_reward}')

            state = env.reset()
            done = False