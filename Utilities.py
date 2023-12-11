import numpy as np
import matplotlib.pyplot as plt

## Utilities
def normalize_observation(obs):
    """
    Returns Normalized Advantages based on the given observation vector.
    """

    ranges = [(0, 1920), (0, 1080), (-np.pi, np.pi), (0, 30), (0, 300), (0, 300), (0, 300), (0, 300), (0, 300)]
    normalized_obs = [(obs[i] - low) / (high - low) for i, (low, high) in enumerate(ranges)]
    return np.array(normalized_obs)

# Discounted Rewards 1
def discount_rewards_1(rewards, gamma=0.99):
    """
    Returns Discounted Rewards based on the given rewards and gamma params.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])   
    return np.array(new_rewards[::-1])

# Discounted Rewards 2
def discount_rewards_2(advantages, values):
    """
    Returns Dicounted Rewards based on advantages and values.
    """

    discounted_returns = advantages + values

    return discounted_returns

def calculate_gaes(rewards, values, gamma=0.99, decay=0.95):
    """
    Returns Advantages.
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])


def plot_learning_curve(x, scores): #figure_file
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    #plt.savefig(figure_file)
