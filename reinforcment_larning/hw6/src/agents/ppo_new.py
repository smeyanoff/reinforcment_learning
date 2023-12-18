import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PpoNew(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.9, batch_size=128,
                 epsilon=0.2, epoch_n=30, pi_lr=1e-4, v_lr=5e-4, lam=0.95):

        super().__init__()

        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 2 * action_dim), nn.Tanh())

        self.v_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 1))

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(
            self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)
        self.lam = lam

    def get_action(self, state):
        mean, log_std = self.pi_model(torch.FloatTensor(state))
        dist = Normal(mean, torch.exp(log_std))
        action = dist.sample()
        return action.numpy().reshape(1)

    def calculate_advantage(self, rewards, states):
        advantages = []
        advantage = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards)-1:
                advantage = rewards[i] - self.gamma * self.v_model(states[i])
            else:
                advantage = rewards[i] + self.gamma * \
                    self.v_model(states[i+1]) - self.v_model(states[i])
            advantages.insert(0, advantage)

        return torch.tensor(advantages, dtype=torch.float32)

    def fit(self, states, actions, rewards, dones):

        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones])
        rewards, dones = rewards.reshape(-1,), dones.reshape(-1, 1)

        states, actions, rewards = map(
            torch.FloatTensor, [states, actions, rewards])

        mean, log_std = self.pi_model(states).T
        mean, log_std = mean.unsqueeze(1), log_std.unsqueeze(1)
        dist = Normal(mean, torch.exp(log_std))
        old_log_probs = dist.log_prob(actions).detach()

        for _ in range(self.epoch_n):

            idxs = np.random.permutation(states.shape[0])
            for i in range(0, states.shape[0], self.batch_size):
                b_idxs = idxs[i:i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = self.calculate_advantage(b_rewards, b_states)

                b_mean, b_log_std = self.pi_model(b_states).T
                b_mean, b_log_std = b_mean.unsqueeze(1), b_log_std.unsqueeze(1)
                b_dist = Normal(b_mean, torch.exp(b_log_std))
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(
                    b_ratio, 1. - self.epsilon,  1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.square(self.v_model(
                    b_states) - b_advantage.unsqueeze(1))

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()
