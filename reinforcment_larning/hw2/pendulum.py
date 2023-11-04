import clearml
import gym
import numpy as np
import optuna
from agent import CEM
from utils import get_elite_trajectories, get_trajectory

gym_name = "pendulum"

env = gym.make('Pendulum-v1', g=9.81)
task = clearml.Task.init(
    project_name="RLearning",
    task_name="HW2_pendulum_6",
)


def objective(trial: optuna.Trial):

    state_dim = 3
    action_n = 1

    epsilon = trial.suggest_float("epsilon", 1e-1, 7e-1, log=True)

    learning_rate = trial.suggest_float("lr", 1e-3, 1, log=True)
    learning_rate_decay = trial.suggest_float(
        "learning_rate_decay",
        1e-2,
        1e-1,
        log=True,
    )
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    hidden_layer_dim = trial.suggest_int("hidden_layer_dim", 10, 60)

    episode_n = trial.suggest_int("n_episodes", 10, 50)
    trajectory_n = trial.suggest_int("n_trajectories", 50, 300)
    trajectory_len = trial.suggest_int("len_trajectories", 500, 1000)
    q_param = trial.suggest_float("q_param", 0.7, 0.99)

    agent = CEM(
        state_dim,
        action_n,
        learning_rate,
        learning_rate_decay,
        weight_decay,
        hidden_layer_dim,
        epsilon,
        gym_name,
    )

    logger = task.get_logger()
    mtr = []

    for episode in range(episode_n):
        trajectories = [
            get_trajectory(env, agent, trajectory_len, trajectory_n)
            for _ in range(trajectory_n)
        ]

        mean_total_reward = np.mean([
            trajectory['total_reward']
            for trajectory in trajectories
        ])

        mtr.append(mean_total_reward)

        # logger.report_text(trajectories[0]["actions"])

        elite_trajectories = get_elite_trajectories(trajectories, q_param)

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)

        logger.report_scalar(
            "mean_total_reward",
            f"trial_{trial.number}",
            mean_total_reward,
            iteration=episode,
        )
        if len(mtr) > 20:
            if np.median(mtr[-10:]) <= np.median(mtr[-20:-10]):
                logger.report_text(
                    f"""
                mtr {mean_total_reward} <= median mtr {np.median(mtr[-10:])}
                    """,
                )
                task.upload_artifact(f"trial_{trial.number}", trial.params)
                return mean_total_reward
    task.upload_artifact(f"trial_{trial.number}", trial.params)

    return mean_total_reward


study = optuna.create_study(directions=["maximize"])
study.optimize(
    objective,
    n_trials=50,
)

task.update_parameters(study.best_params)

task.close()
