import numpy as np


def get_trajectory(env, agent, trajectory_len, visualize=False):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}

    state = env.reset()[0]

    for _ in range(trajectory_len):

        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, wtf1, wtf2 = env.step(action)
        trajectory['total_reward'] += reward

        if done:
            break

        if visualize:
            env.render()

    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [
        trajectory for trajectory in trajectories
        if trajectory['total_reward'] > quantile
    ]
