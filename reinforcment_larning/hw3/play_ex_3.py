import clearml
import optuna
from agent_value_iteration import Agent
from Frozen_Lake import FrozenLakeEnv
from numpy import column_stack, mean
from numpy.random import choice

if __name__ == "__main__":

    logging = True

    def objective(trial: optuna.Trial):
        iter_n = trial.suggest_int("n_iterations", 500, 2000)

        gamma = trial.suggest_float("gamma", 0.8, 1)
        epsilon = trial.suggest_float("epsilon", 1e-5, 5e-2)

        env = FrozenLakeEnv()
        agent = Agent(
            gamma=gamma,
            env=env,
            epsilon=epsilon,
            early_stop=True
        )

        agent.fit(iter_n)

        total_rewards = []

        for _ in range(1000):
            total_reward = 0
            state = env.reset()
            for _ in range(1000):
                action = choice(
                    env.get_possible_actions(state),
                    p=list(agent.policy[state].values()),
                )
                state, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    break
            total_rewards.append(total_reward)

        return mean(total_rewards)

    def callback(
        gamma_list: list,
        rewards_list: list,
        _study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ):

        gamma_list.append(trial.params["gamma"])
        rewards_list.append(trial.value)

    gamma_list = []
    rewards_list = []

    study = optuna.create_study(directions=["maximize"])
    study.optimize(
        objective,
        n_trials=200,
        callbacks=[lambda x, y: callback(gamma_list, rewards_list, x, y)],
    )

    if logging:

        task = clearml.Task.init(
            project_name="RLearning",
            task_name="HW3_ex_3_val_iter",
        )

        logger = clearml.Logger.current_logger()

        plot_data = column_stack((rewards_list, gamma_list))

        logger.report_scatter2d(
            title="actor_from_gamma",
            series="reward_from_gamma",
            mode="markers",
            iteration=0,
            scatter=plot_data,
            xaxis="gamma",
            yaxis="reward",
        )
