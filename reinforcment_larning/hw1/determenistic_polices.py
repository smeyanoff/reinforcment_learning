import itertools

import gym
import numpy as np
import yaml
from clearml import Task

with open("config.yaml", "r") as conf:
    config = yaml.safe_load(conf)
config = config["hw1"]["smoothing"]


class Envitonment:
    def __init__(self):
        self.env = gym.make('Taxi-v3')

    def get_trajectory(self):
        obs = self.env.reset()

        done = False
        state = obs[0]
        available_steps = obs[1]

        set_polices = list(set(itertools.permutations([0, 0, 0, 0, 0, 1])))
        probabilities = np.ones((500, 6)) * [0, 0, 0, 0, 0, 1]

        trajectories = []
        for policy in set_polices:
            for action in range(len(probabilities)):
                probabilities_copy = probabilities.copy()
                probabilities_copy[action] = policy
                trajectory = {'states': [], 'actions': [], 'reward': []}
                for _ in range(1000):
                    if done:
                        break
                    next_step = list(probabilities[state]).index(1)

                    trajectory['actions'].append(next_step)

                    obs = self.env.step(next_step)
                    state, reward, done, _, available_steps = obs

                    trajectory['reward'].append(reward)
                    trajectory['states'].append(state)
                trajectories.append(trajectory)

        return trajectories


class CrossEntropyActor:

    def __init__(self):
        self.__observation_space = 500
        self.__action_space = 6

        # probabilities
        self.model = (
            np.ones((self.__observation_space, self.__action_space))
            / self.__action_space
        )

    def choose_action(self, probabilities: list):

        action = np.random.choice(
            list(range(self.__action_space)),
            p=probabilities,
        )
        return action

    def make_step(
        self,
        state: int,
        available_steps: int,
    ):
        step = self.choose_action(self.model[state])
        return step

    def fit(
        self,
        elite_trajectories: list[dict],
        lambd: float = 1,
        smoothing: str = "none",
    ):
        new_model = np.zeros((self.__observation_space, self.__action_space))

        for trajectory in elite_trajectories:
            for state, action in zip(
                trajectory['states'],
                trajectory['actions'],
            ):
                new_model[state][action] += 1

        for state in range(self.__observation_space):
            if smoothing == "none":
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = self.model[state].copy()
            if smoothing == "laplace":
                new_model[state] = ((new_model[state] + lambd) /
                                    (
                                        np.sum(new_model[state]) +
                                        self.__action_space*lambd
                                    ))
            if smoothing == "policy":
                new_model[state] = (
                    new_model[state]*lambd +
                    (1-lambd)*self.model[state]
                )
                new_model[state] /= np.sum(new_model[state])

        self.model = new_model
        return None

        self.model = new_model
        return None


if __name__ == "__main__":
    q_param = config["params"]["q_param"]
    iteration_n = config["params"]["n_iterations"]
    trajectory_n = config["params"]["n_trajectories"]

    env_class = Envitonment()
    actor = CrossEntropyActor()

    if config["params"]["log"]:
        task_name = config["task_name"] + \
            "_".join([str(x) for x in config["params"].values()])
        task = Task.init(
            project_name='RLearning',
            task_name=task_name,
        )
        task.connect(config["params"])

        logger = task.get_logger()

    for iteration in range(iteration_n):
        print("iteration:", iteration)

        trajectories = env_class.get_trajectory()
        # policy evaluation
        total_rewards = [
            np.sum(trajectory['reward'])
            for trajectory in trajectories
        ]
        if config["params"]["log"]:
            logger.report_scalar(
                title='mean total reward',
                series='mtr',
                value=np.mean(total_rewards),
                iteration=iteration,
            )

        # policy improvement
        quantile = np.quantile(total_rewards, q_param)
        if not config["params"]["log"]:
            print(quantile)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['reward'])
            if not config["params"]["log"]:
                print(total_reward)
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        actor.fit(
            elite_trajectories,
            config["params"]["lambda_param"],
            config["params"]["smoothing_type"],
        )
