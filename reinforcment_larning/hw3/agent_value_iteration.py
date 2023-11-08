import copy

import numpy as np


class Agent:

    def __init__(
        self,
        gamma,
        env,
        epsilon
    ):
        self.gamma = gamma
        self.env = env
        self.epsilon = epsilon

        self.v_values = self.init_v_values()
        self.policy = self.init_policy()

    def init_policy(self):
        policy = {}
        for state in self.env.get_all_states():
            policy[state] = {}
            for action in self.env.get_possible_actions(state):
                policy[state][action] = 1 / \
                    len(self.env.get_possible_actions(state))
        return policy

    def init_v_values(self):
        v_values = {}
        for state in self.env.get_all_states():
            v_values[state] = 0
        return v_values

    def p_v_sum(self, state, action):

        value = 0
        for next_state in self.env.get_next_states(state, action):
            value += self.env.get_reward(state, action, next_state)
            value += (
                self.gamma
                * self.env.get_transition_prob(state, action, next_state)
                * self.v_values[next_state]
            )
        return value

    def value_iteration(self):

        new_v_values = copy.deepcopy(self.v_values)
        for state in self.env.get_all_states():
            max_q_values = [
                self.p_v_sum(state, action)
                for action in self.env.get_possible_actions(state)
            ]
            if max_q_values:
                new_v_values[state] = np.max(max_q_values)

        return new_v_values

    def get_q_values(self):
        q_values = {}
        for state in self.env.get_all_states():
            q_values[state] = {}
            for action in self.env.get_possible_actions(state):
                q_values[state][action] = 0
                for next_state in self.env.get_next_states(state, action):
                    if self.env.is_terminal(next_state):
                        next_value = 0
                    else:
                        next_value = self.v_values[next_state]

                    q_values[state][action] += (
                        self.env.get_reward(state, action, next_state)
                    )
                    q_values[state][action] += (
                        self.gamma
                        * self.env.get_transition_prob(state, action, next_state)
                        * next_value
                    )

        return q_values

    def policy_improvement(self):
        policy = {}
        for state in self.env.get_all_states():
            policy[state] = {}
            argmax_action = None
            max_q_value = float('-inf')
            for action in self.env.get_possible_actions(state):
                policy[state][action] = 0
                if self.q_values[state][action] > max_q_value:
                    argmax_action = action
                    max_q_value = self.q_values[state][action]
            policy[state][argmax_action] = 1

        self.policy = policy

        return None

    def fit(self, iter_n):
        for _ in range(iter_n):
            new_v_values = self.value_iteration()
            if np.max(
                np.abs(
                    np.subtract(
                        list(new_v_values.values()),
                        list(self.v_values.values())
                    )
                )
            ) < self.epsilon:
                break
            self.v_values = new_v_values

        self.q_values = self.get_q_values()
        self.policy_improvement()
