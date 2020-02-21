import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from gym import make
import pybullet_envs


ENV_NAME = 'AntBulletEnv-v0'

HIDDEN_SIZE = 256
HIDDEN_LAYERS = 2

BUFFER_CAPACITY = 100000
GAMMA = 0.99
EXPL_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
UPDATE_TARGET_TAU = 0.005
MAX_STEPS_PER_EPISODE = 1005
EPISODE_START = 50

BATCH_SIZE = 256
LR = 0.0003

EVAL_EPISODES_CNT = 50

MODEL_PATH = 'model.pth'


def phi(state):
    #_ = state
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


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        assert HIDDEN_LAYERS > 0

        layers = [nn.Linear(in_dim, HIDDEN_SIZE),
                  nn.ReLU()]
        for _ in range(HIDDEN_LAYERS - 1):
            layers += [nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                       nn.ReLU()]
        layers += [nn.Linear(HIDDEN_SIZE, out_dim)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.model(state)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.model = nn.Sequential(MLP(state_dim, action_dim),
                                   nn.Tanh())

        self.max_action = max_action
        
    def forward(self, state):
        return self.max_action * self.model(state)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.model = MLP(state_dim + action_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.model(x)


class A2C:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.actor.eval()
        self.actor_target.eval()

        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = optim.Adam(
            [p for m in [self.critic1, self.critic2] for p in m.parameters()],
            lr=LR
        )
        self.critic1.eval()
        self.critic2.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

        self.criterion = nn.MSELoss()

        self.max_action = max_action

        self.step = 0

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor([state])
            action = self.actor(state)
            return action.numpy()[0]

    def soft_update(self, model_target, model):
        for tp, lp in zip(model_target.parameters(), model.parameters()):
            tp.data.copy_(UPDATE_TARGET_TAU * lp.data + (1.0 - UPDATE_TARGET_TAU) * tp.data)

    def update(self, batch):
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        with torch.no_grad():
            noise = torch.randn_like(action) * POLICY_NOISE * self.max_action
            noise = noise.clamp(-NOISE_CLIP * self.max_action, 
                                 NOISE_CLIP * self.max_action)

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target)
            q_target[done] = 0
        q_target = reward.reshape(-1, 1) + q_target * GAMMA

        self.critic1.train()
        self.critic2.train()

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        critic_loss = self.criterion(q1, q_target) + self.criterion(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic1.eval()
        self.critic2.eval()

        if self.step % POLICY_FREQ == 0:
            self.actor.train()

            actor_loss = -self.critic1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.actor.eval()

            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)
            self.soft_update(self.actor_target, self.actor)

        self.step += 1

    def save(self, path):
        self.actor.save(path)


if __name__ == '__main__':
    env = make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    a2c = A2C(state_dim, action_dim, max_action)

    buffer = Buffer(BUFFER_CAPACITY)

    episode = 0
    all_rewards = []
    while True:
        total_reward = 0
        state = env.reset()
        for _ in range(MAX_STEPS_PER_EPISODE):
            if episode < EPISODE_START:
                action = env.action_space.sample()
            else:
                action = a2c.act(state)
                action = action + np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
                action = action.clip(-max_action, max_action)

            new_state, reward, done, _ = env.step(action)
            modified_reward = reward + (GAMMA * phi(new_state) - phi(state))
            total_reward += reward

            buffer.add((state, action, modified_reward, new_state, done))

            state = new_state

            if episode >= EPISODE_START:
                a2c.update(buffer.sample(BATCH_SIZE))

            if done:
                break

        a2c.save(MODEL_PATH)

        all_rewards += [total_reward]
        last_rewards = all_rewards[-EVAL_EPISODES_CNT:]
        last_mean_reward = sum(last_rewards) / len(last_rewards)

        print('episode={}   reward={:.4}   mean reward={:.4}'.format(
                episode, total_reward, last_mean_reward))

        episode += 1