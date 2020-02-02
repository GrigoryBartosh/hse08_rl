import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from gym import make


ENV_NAME = 'LunarLander-v2'
STATE_DIM = 8
ACTION_DIM = 4

GAMMA = 0.99
EPSILON_START = 0.95
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_CAPACITY = 100000

HIDDEN_SIZE = 256
HIDDEN_LAYERS = 2

BATCH_SIZE = 64
LR = 0.0005

MAX_STEPS_PER_EPISODE = 1000
UPDATE_TARGET_TAU = 0.001
EVAL_EPISODES_CNT = 50

MODEL_PATH = 'model.pth'


def phi(state):
    x, y, vx, vy, a, va, c1, c2 = state
    loss = 0
    return -loss


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
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE, hidden_layers=HIDDEN_LAYERS):
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

    def soft_update(self):
        for tp, lp in zip(self.q_foo_target.parameters(), self.q_foo.parameters()):
            tp.data.copy_(UPDATE_TARGET_TAU * lp.data + (1.0 - UPDATE_TARGET_TAU) * tp.data)

    def update(self, batch):
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        self.q_foo.train()

        with torch.no_grad():
            q_target = self.q_foo_target(next_state).max(dim=1)[0]
            q_target[done] = 0
        q_target = reward + q_target * GAMMA

        q = self.q_foo(state).gather(1, action[:, None]).squeeze()

        loss = self.criterion(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.q_foo.eval()

        self.soft_update()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            action = self.q_foo(state).argmax(dim=1)
            return action.numpy()[0]

    def update_target(self):
        self.q_foo_target = copy.deepcopy(self.q_foo)

    def save(self, path):
        self.q_foo.save(path)


if __name__ == '__main__':
    env = make(ENV_NAME)

    aql = AQL(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    buffer = Buffer(BUFFER_CAPACITY)

    epsilon = EPSILON_START
    step = 0
    episode = 0
    all_rewards = []
    while True:
        total_reward = 0
        state = env.reset()
        for _ in range(MAX_STEPS_PER_EPISODE):
            if np.random.rand() < epsilon:
                action = np.random.randint(ACTION_DIM)
            else:
                action = aql.act(state)

            new_state, reward, done, _ = env.step(action)
            modified_reward = reward + (GAMMA * phi(new_state) - phi(state))
            total_reward += reward

            buffer.add((state, action, modified_reward, new_state, done))

            step += 1
            state = new_state

            if step > BATCH_SIZE:
                aql.update(buffer.sample(BATCH_SIZE))

            if done:
                break

        aql.save(MODEL_PATH)

        all_rewards += [total_reward]
        last_rewards = all_rewards[-EVAL_EPISODES_CNT:]
        last_mean_reward = sum(last_rewards) / len(last_rewards)

        print('episode={}   eps={:.4}   reward={:.4}   mean reward={:.4}'.format(
                episode, epsilon, total_reward, last_mean_reward))        

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode += 1