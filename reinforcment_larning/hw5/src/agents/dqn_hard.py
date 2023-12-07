import random

import numpy as np
import torch
from src.functions.q_function import Qfunction


class DQNHARD:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        epsilon_decrease=0.01,
        epilon_min=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def update_memory(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

    def get_targets(self, rewards, dones, next_states, q_function=None):

        assert len(self.memory) > self.batch_size

        if q_function is None:
            q_function = self.q_function

        targets = rewards + self.gamma * (1 - dones) * torch.max(
            q_function(next_states),
            dim=1,
        ).values

        return targets

    def get_batch(self):

        assert len(self.memory) > self.batch_size

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = map(
            torch.tensor, list(zip(*batch)),
        )

        return states, actions, rewards, dones, next_states

    def get_qvalues(self, states, actions, q_function=None):

        assert len(self.memory) > self.batch_size

        if q_function is None:
            q_function = self.q_function

        q_values = q_function(states)[torch.arange(self.batch_size), actions]

        return q_values

    def fit(self, q_function=None):

        states, actions, rewards, dones, next_states = self.get_batch()

        targets = self.get_targets(rewards, dones, next_states)

        q_values = self.get_qvalues(states, actions, q_function)

        loss = torch.mean((q_values - targets.detach()) ** 2)
        loss.backward()
        self.optimzaer.step()
        self.optimzaer.zero_grad()

        if self.epsilon - self.epsilon_decrease > self.epilon_min:
            self.epsilon -= self.epsilon_decrease
