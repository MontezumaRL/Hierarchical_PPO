from config import EPISODE_SIZE, CLIP_EPSILON, WIDTH, HEIGHT
from models.policy_model import PolicyModel
from models.value_function_model import ValueFunction
from preprocessing.preprocess import preprocess_observation
from frame_stack import FrameStack
from reward import get_reward
import torch
import gymnasium as gym
import ale_py

def train():
    num_epochs = 100
    old_policy = PolicyModel()
    new_policy = PolicyModel()
    new_policy.copy_weights_from(old_policy)
    value_function = ValueFunction()
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")
    framestack = FrameStack()

    # For observation resizing
    downscale = 0.9
    new_height, new_width = (int(HEIGHT*downscale),int(WIDTH*downscale))

    observation, info = env.reset()
    obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
    framestack.reset(obs)
    data_per_epochs = []

    for epoch in range(num_epochs):
        
        advantage_values = []
        old_policy_values = []
        new_policy_values = []
        rewards = []
        reward = 0

        for episode in range(EPISODE_SIZE):

            # Evaluate the state :
            stacked_observation = framestack.get_stacked_frames_tensor()

            # Calculate rewards and advantage estimation
            custom_reward = get_reward(stacked_observation)
            full_reward = torch.tensor([custom_reward + reward])
            value_estimate = value_function.forward(stacked_observation)
            advantage_estimate = full_reward - value_estimate

            # Calculation actions probability
            action_probabilities = new_policy.forward(stacked_observation)
            old_action_probabilities = old_policy.forward(stacked_observation)

            # Doing the best action
            action = torch.argmax(action_probabilities, dim=1)
            observation, reward, done, truncated, info = env.step(action)
            processed_obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
            framestack.add_frame(processed_obs)

            # Récupérer les données de l'episode
            advantage_values.append(advantage_estimate)
            old_policy_values.append(old_action_probabilities)
            new_policy_values.append(action_probabilities)
            rewards.append(full_reward)

        data_per_epochs.append([advantage_values, old_policy_values, new_policy_values, rewards])

        # 

train()