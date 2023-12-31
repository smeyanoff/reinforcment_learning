import clearml
import gym
import numpy as np
import optuna
from agent import CEM
from utils import get_elite_trajectories, get_trajectory

gym_name = "lunar_lander"

env = gym.make(
    id="LunarLander-v2",
    gravity=-10.0,
    wind_power=15.0,
    turbulence_power=1.5,
    continuous=False,
    enable_wind=False,
)
task = clearml.Task.init(
    project_name="RLearning",
    task_name="HW2_lunar_lender_3",
)


def objective(trial: optuna.Trial):

    state_dim = 8
    action_n = 4
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    hidden_layer_dim = trial.suggest_int("hidden_layer_dim", 50, 200)

    episode_n = trial.suggest_int("n_episodes", 10, 100)
    trajectory_n = trial.suggest_int("n_trajectories", 50, 200)
    trajectory_len = trial.suggest_int("len_trajectories", 500, 2000)
    q_param = trial.suggest_float("q_param", 0.8, 0.99)

    agent = CEM(state_dim, action_n, learning_rate, hidden_layer_dim, gym_name)

    logger = task.get_logger()

    for episode in range(episode_n):
        trajectories = [
            get_trajectory(env, agent, trajectory_len, trajectory_n)
            for _ in range(trajectory_n)
        ]

        mean_total_reward = np.mean([
            trajectory['total_reward']
            for trajectory in trajectories
        ])

        elite_trajectories = get_elite_trajectories(trajectories, q_param)

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)

        logger.report_scalar(
            "mean_total_reward",
            f"trial_{trial.number}",
            mean_total_reward,
            iteration=episode,
        )
    task.upload_artifact(f"trial_{trial.number}", trial.params)

    return mean_total_reward


study = optuna.create_study(directions=["maximize"])
study.optimize(
    objective,
    n_trials=50,
)

task.update_parameters(study.best_params)

task.close()
