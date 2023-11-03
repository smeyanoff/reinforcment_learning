import numpy as np
import torch
from torch import nn


class CEM(nn.Module):
    def __init__(
        self,
        state_dim,
        action_n,
        learning_rate,
        learning_rate_decay,
        weight_decay,
        hidden_layer_dim,
        task,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, self.action_n),
        )

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.task = task
        if self.task == "lunar_lander":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
            )
            self.loss = nn.CrossEntropyLoss()
        elif self.task == "pendulum":
            self.optimizer = torch.optim.Adagrad(
                self.parameters(),
                lr=learning_rate,
                lr_decay=0.01,
                weight_decay=0.01,
            )
            self.loss = nn.L1Loss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)

        if self.task == "lunar_lander":
            action_prob = self.softmax(logits).detach().numpy()
            action = np.random.choice(self.action_n, p=action_prob)
        elif self.task == "pendulum":
            action = torch.clamp(logits, -2, 2).detach().numpy()
            # action = self.tanh(logits).detach().numpy() * 2

        return action

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []

        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])

        elite_states = torch.FloatTensor(elite_states)
        if self.task == "lunar_lander":
            elite_actions = torch.LongTensor(elite_actions)
        elif self.task == "pendulum":
            elite_actions = torch.FloatTensor(elite_actions)

        loss = self.loss(self.forward(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
