import gym
import numpy as np
from clearml import Task
import yaml

with open("config.yaml", "r") as conf:
    config = yaml.safe_load(conf)
config = config["hw1"]["smoothing"]


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

            next_step = agent.make_step(state, available_steps['action_mask'])
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
        self.model = (np.ones((self.__observation_space, self.__action_space))
                      / self.__action_space)

    def choose_action(self, probabilities: list):

        p = [x / np.sum(probabilities) for x in probabilities]
        action = np.random.choice(list(range(self.__action_space)), 
                                  p=p)
        return action
    
    def make_step(self, state: int, available_steps:int):
        probabilities = self.model[state] * available_steps
        if np.sum(probabilities) > 0:
            step = self.choose_action(probabilities)
        else:
            step = self.choose_action(self.model[state])

        return step

    def fit(self, elite_trajectories: list[dict], smoothing="none", l=1):
        assert smoothing in ["none", "policy", "laplace"]
        if smoothing == "laplace":
            assert l > 0
        if smoothing == "policy":
            assert l > 0 and l <= 1
        new_model = np.zeros((self.__observation_space, self.__action_space))

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1


        if smoothing == "none":
            for state in range(self.__observation_space):
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = self.model[state].copy()
        if smoothing == "laplace":
            for state in range(self.__observation_space):
                new_model[state] = ((new_model[state] + l) / 
                                    (np.sum(new_model[state])+self.__action_space*l))
        if smoothing == "policy":
            for state in range(self.__observation_space):
                if np.sum(new_model[state]) > 0:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = self.model[state].copy()
                new_model[state] = (new_model[state]*l + (1-l)*self.model[state])

        self.model = new_model
        return None


if __name__=="__main__":
    q_param = config["params"]["q_param"]
    iteration_n = config["params"]["n_iterations"]
    trajectory_n = config["params"]["n_trajectories"]

    env_class = Envitonment()
    actor = CrossEntropyActor()

    task_name = (
        config["task_name"] 
        + "_".join([str(x) for x in config["params"].values()])
        + "_".join([str(x) for x in config["fit_params"].values()])
    )
    task = Task.init(project_name='RLearning', 
                     task_name=task_name)
    task.connect(config)

    logger = task.get_logger()

    for iteration in range(iteration_n):
        trajectories = [env_class.get_trajectory(actor) for _ in range(trajectory_n)]
        #policy evaluation
        total_rewards = [np.mean(trajectory['reward']) for trajectory in trajectories]
        logger.report_scalar(title='mean total reward', 
                             series='mtr', 
                             value=np.mean(total_rewards),
                             iteration=iteration)

        #policy improvement
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['reward'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)
        
        actor.fit(elite_trajectories, 
                  config["fit_params"]["smoothing_type"],
                  config["fit_params"]["lambda_param"])
