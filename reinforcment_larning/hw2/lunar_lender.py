import gym
import numpy as np
from agent import CEM
from utils import get_elite_trajectories, get_trajectory

env = gym.make(
    id="LunarLander-v2",
    gravity=-10.0,
    wind_power=15.0,
    turbulence_power=1.5,
    continuous=False,
    enable_wind=False,
)

state_dim = 8
action_n = 4
learning_rate = 0.02
hidden_layer_dim = 128

agent = CEM(state_dim, action_n, learning_rate, hidden_layer_dim)
episode_n = 50
trajectory_n = 100
trajectory_len = 50
q_param = 0.95

for episode in range(episode_n):
    trajectories = [
        get_trajectory(env, agent, trajectory_n)
        for _ in range(trajectory_n)
    ]

    mean_total_reward = np.mean([
        trajectory['total_reward']
        for trajectory in trajectories
    ])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')

    elite_trajectories = get_elite_trajectories(trajectories, q_param)

    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
