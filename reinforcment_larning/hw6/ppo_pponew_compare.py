import gym
from src.agents import PPO, PpoNew

if __name__ == "__main__":

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PpoNew(state_dim, action_dim)

    episode_n = 50
    trajectory_n = 20
    done = False

    total_rewards = []

    for episode in range(episode_n):

        states, actions, rewards, log_probs = [], [], [], []

        for _ in range(trajectory_n):
            total_reward = 0

            state = env.reset()
            while not done:
                states.append(state)

                action, log_prob = agent.get_action(state)
                actions.append(action)
                log_probs.append(log_prob)

                state, reward, done, _ = env.step(2 * action)
                rewards.append(reward)

                total_reward += reward

            total_rewards.append(total_reward)
            print(total_reward)

        agent.fit(states, actions, rewards, log_probs)
