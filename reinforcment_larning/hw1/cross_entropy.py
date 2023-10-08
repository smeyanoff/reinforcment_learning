import gym
import numpy as np
import yaml
from clearml import Task

with open("config.yaml", "r") as conf:
    config = yaml.safe_load(conf)
config = config["hw1"]["cross_entropy"]


class Envitonment:
    def __init__(self):
        self.env = gym.make('Taxi-v3')

    def get_trajectory(self, agent, max_steps=1000):
        obs = self.env.reset()
        trajectory = {'states': [], 'actions': [], 'reward': []}
        done = False
        state = obs[0]
        available_steps = obs[1]

        for _ in range(max_steps):

            if done:
                break

            next_step = agent.make_step(
                state,
                available_steps['action_mask'],
            )
            trajectory['actions'].append(next_step)

            obs = self.env.step(next_step)
            state, reward, done, _, available_steps = obs

            trajectory['reward'].append(reward)
            trajectory['states'].append(state)

        return trajectory


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

    def fit(self, elite_trajectories: list[dict]):
        new_model = np.zeros((self.__observation_space, self.__action_space))

        for trajectory in elite_trajectories:
            for state, action in zip(
                trajectory['states'],
                trajectory['actions'],
            ):
                new_model[state][action] += 1

        for state in range(self.__observation_space):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

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
        trajectories = [
            env_class.get_trajectory(
                actor,
            ) for _ in range(trajectory_n)
        ]
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

        actor.fit(elite_trajectories)
