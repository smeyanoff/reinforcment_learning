import clearml
import optuna
from agent import Agent as Agent_p_iter
from agent_value_iteration import Agent as Agent_v_iter
from Frozen_Lake import FrozenLakeEnv
from numpy import column_stack, mean
from numpy.random import choice

if __name__ == "__main__":

    logging = True

    if logging:

        task = clearml.Task.init(
            project_name="RLearning",
            task_name="HW3_agents_compare",
        )

        logger = clearml.Logger.current_logger()

    env = FrozenLakeEnv()

    gamma = 0.997
    eval_iter_n = 100
    iter_n = 100
    epsilon = 0.005

    agent_v_iter = Agent_v_iter(
        gamma=gamma,
        env=env,
        epsilon=epsilon,
        early_stop=False
    )
    agent_p_iter_warm_false = Agent_p_iter(
        gamma=gamma,
        env=env,
        eval_iter_n=eval_iter_n,
        warmstart=False
    )
    agent_p_iter_warm_true = Agent_p_iter(
        gamma=gamma,
        env=env,
        eval_iter_n=eval_iter_n,
        warmstart=True
    )

    agent_info_dict = {}

    for agent_name, agent in zip(
        ("Agent_Val_iter", "Agent_P_iter_warmstart_false", "Agent_P_iter_warmstart_True"),
        (agent_v_iter, agent_p_iter_warm_false, agent_p_iter_warm_true)
    ):
        count_action = []
        total_rewards = []
        agent_info_dict[agent_v_iter] = {}
        for _ in range(100):
            agent.fit(iter_n)
            total_reward = 0
            state = env.reset()
            for _ in range(100):
                action = choice(
                    env.get_possible_actions(state),
                    p=list(agent.policy[state].values()),
                )
                state, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    break
            total_rewards.append(total_reward)
            count_action.append(agent.count_trigger_env)

        if logging:
            plot_data = column_stack(
                (mean(count_action), mean(total_rewards))
            )

            logger.report_scatter2d(
                title="actor_from_actions",
                series=agent_name,
                mode="markers",
                iteration=0,
                scatter=plot_data,
                xaxis="mean_count_action",
                yaxis="mean_count_reward",
            )
