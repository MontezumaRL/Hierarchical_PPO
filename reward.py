from math import sqrt

def get_reward(num_lives):
    reward_per_life = {6:100, 5:95, 4:90, 3:80, 2:35, 1:5, 0:0}
    life_reward = reward_per_life[num_lives]
    return life_reward

def calculate_discounted_rewards(rewards, gamma=0.6, future_rewards_count=8):
    """
    Calculate the discounted rewards with the future rewards added to the reward at time t.
    """
    discounted_rewards = []
    episode_length = len(rewards)

    for t in range(episode_length):
        future_rewards = 0
        # Sum future rewards with the discount factor gamma^k
        for k in range(1, future_rewards_count + 1):
            if t + k < episode_length:  # Ensure we don't go out of bounds
                future_rewards += rewards[t + k] * (gamma ** k)
            else:
                future_rewards += rewards[t] * (gamma ** k)
        # Current reward + discounted future rewards
        discounted_rewards.append(rewards[t] + future_rewards)

    return discounted_rewards