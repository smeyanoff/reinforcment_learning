import clearml
import gym
import numpy as np
import optuna
from src.agents.dqn_hard import DQNHARD


def objective(trial: optuna.Trial):

    batch_size = trial.suggest_int("batch_size", 100, 250)
    epsilon_decrease = trial.suggest_float("epsilon_decrease", 1e-5, 1e-1)
    epilon_min = trial.suggest_float("epilon_min", 1e-5, 1e-1)

    update_frequency = trial.suggest_int("update_frequency", 2, 15)

    gamma = trial.suggest_float("gamma", 0.7, 1)
    lr = trial.suggest_float("lr", 1e-5, 1e-1)

    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Создание основной и целевой нейронных сетей
    main = DQNHARD(
        state_dim,
        action_dim,
        gamma,
        lr,
        batch_size,
        epsilon_decrease,
        epilon_min,
    )

    target = DQNHARD(
        state_dim,
        action_dim,
        gamma,
        lr,
        batch_size,
        epsilon_decrease,
        epilon_min,
    )

    # Копирование весов основной сети в целевую
    target.q_function.load_state_dict(main.q_function.state_dict())

    episode_n = 100
    t_max = 600

    total_rewards = []

    for _ in range(episode_n):
        total_reward = 0

        state = env.reset()[0]
        for step in range(t_max):
            # Выбор действия на основе Q-значений основной сети
            action = main.get_action(state)

            next_state, reward, done, _, _ = env.step(action)

            main.update_memory(state, action, reward, done, next_state)

            if step > batch_size:

                total_reward += reward

                # Рассчитываем Q-значения для текущего
                # состояния с использованием основной сети
                # Расчет целевого Q-значения с использованием целевой сети
                main.fit(target.q_function)

                # Периодическое обновление весов целевой сети
                if step % update_frequency == 0:
                    target.q_function.load_state_dict(
                        main.q_function.state_dict(),
                    )

                state = next_state

                if done:
                    break

        total_rewards.append(total_reward)

    global TOTAL_REW
    TOTAL_REW = total_rewards

    return np.mean(total_rewards)


def callback(
    logger: clearml.Logger | None,
    _study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
):
    if logger is not None:
        logger.report_scalar(
            "mean_total_reward",
            f"trial_{trial.number}",
            trial.value,
            iteration=trial.number,
        )
        task.upload_artifact(f"trial_{trial.number}", trial.params)

        plot_data = np.column_stack((
                                    list(range(100)),
                                    np.array(TOTAL_REW),
                                    ))

        logger.report_scatter2d(
            title="iteration_reward",
            series=f"trial_{trial.number}",
            iteration=0,
            scatter=plot_data,
            xaxis="iteration",
            yaxis="reward",
        )


if __name__ == "__main__":

    logging = True
    task_name = "HW5_LunarLender_2_hard"

    logger = None

    if logging:

        task = clearml.Task.init(
            project_name="RLearning",
            task_name=task_name,
        )

        logger = clearml.Logger.current_logger()

    study = optuna.create_study(directions=["maximize"])
    study.optimize(
        objective,
        n_trials=50,
        callbacks=[lambda study, trial: callback(logger, study, trial)],
    )
