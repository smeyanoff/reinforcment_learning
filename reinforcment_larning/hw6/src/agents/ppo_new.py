import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PpoNew(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma=0.9,
                 batch_size=128,
                 epsilon=0.2,
                 epoch_n=30,
                 pi_lr=1e-4,
                 v_lr=5e-4):

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

    def get_action(self, state):

        state_tensor = torch.FloatTensor(state)
        mean, log_std = self.pi_model(state_tensor)
        dist = Normal(mean, torch.exp(log_std))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def calculate_advantage(self, rewards, perm_log_probs):
        advantages = []
        advantage = 0
        next_advantage = 0

        for i in reversed(range(len(rewards)))[:-1]:
            delta = rewards[i] + self.gamma * torch.exp(perm_log_probs[i]) \
                - torch.exp(values[i-1])
            advantage = delta + self.gamma * next_advantage
            advantages.insert(0, advantage)
            next_advantage = advantage

        advantages.insert(0, 0.)

        return torch.tensor(advantages, dtype=torch.float32)

    def fit(self, states, rewards, log_probs):

        lens = len(states)

        states, rewards, log_probs = map(
            torch.FloatTensor, [states, rewards, log_probs]
        )

        for _ in range(self.epoch_n):
            idxs = np.random.permutation(lens)
            for i in range(0, lens, self.batch_size):
                idxs_i = idxs[i:i + self.batch_size]
                perm_states, perm_rewards, perm_log_probs \
                    = map(T[idxs_i], [states, rewards, log_probs])

                advantages = calculate_advantage(perm_rewards, perm_log_probs)
                advantages = (advantages.detach() - advantages.mean()) \
                    / (advantages.std() + 1e-8)

                action_values = perm_log_probs * advantages.detach()
                clipped_action_values = log_probs * \
                    torch.clamp(advantages, 1 - clip_epsilon, 1 + clip_epsilon)
                policy_loss = -torch.min(action_values, clipped_action_values)

                policy_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                b_advantage = advantages.detach() - self.v_model(perm_states)

                v_loss = torch.mean(b_advantage ** 2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()
