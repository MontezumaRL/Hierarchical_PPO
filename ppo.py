from config import EPISODE_SIZE, WIDTH, HEIGHT
from models.policy_model import PolicyModel
from models.value_function_model import ValueFunction
from preprocessing.preprocess import preprocess_observation, rgbToGray
from frame_stack import FrameStack
from reward import get_reward, calculate_discounted_rewards
import torch
import torch.nn.functional as F
import gymnasium as gym
import ale_py
from random import randint
from loss import ppo_policy_loss
from rnd import PredictorNet, TargetNet
import argparse
    
def train(num_epochs=30, policy_model=None, value_model=None, save_path_policy=None, save_path_value=None):
    params = [policy_model, value_model, save_path_policy, save_path_value]
    for i,param in enumerate(params):
        if param == "":
            params[i] = None
    if num_epochs == "":
        num_epochs = 30
    else:
        num_epochs = int(num_epochs)
    print(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Init of necessary objects
    old_policy = PolicyModel()
    new_policy = PolicyModel()
    old_policy.to(device)
    new_policy.to(device)
    if params[0] != None:
        print("Loading policy network\n")
        old_policy.load_state_dict(torch.load(policy_model))
        old_policy.eval()

    new_policy.copy_weights_from(old_policy)
    new_policy.eval()
    value_function = ValueFunction()
    value_function.to(device)
    if params[1] != None:
        print("Loading value network\n")
        value_function.load_state_dict(torch.load(value_model))
        value_function.eval()
    predictor_net = PredictorNet(input_dim=WIDTH * HEIGHT)
    predictor_net.to(device)
    target_net = TargetNet(input_dim=WIDTH * HEIGHT)
    target_net.to(device)
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="rgb_array") #human or rgb_array
    framestack = FrameStack()
    optimizer_policy = torch.optim.Adam(new_policy.parameters(), lr=0.001)
    optimizer_value = torch.optim.Adam(value_function.parameters(), lr=0.003)
    predictor_optimizer = torch.optim.Adam(predictor_net.parameters(), lr=0.008)
    predictor_optimizer_late = torch.optim.Adam(predictor_net.parameters(), lr=0.0001)

    for param in target_net.parameters():
        param.requires_grad = False
    # For observation resizing
    downscale = 0.9
    new_height, new_width = (int(HEIGHT*downscale),int(WIDTH*downscale))
    observation, info = env.reset()
    processed_obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
    framestack.reset(processed_obs)

    data_per_epochs = []
    list_loss = []
    rewards_list = []
    for epoch in range(num_epochs):
        print(f"-----------------------------EPOCH {epoch} --------------------------------------------")
        new_policy.train()
        value_function.train()
        old_policy.eval()

        value_function_values = []
        old_policy_values = []
        new_policy_values = []
        rewards = []
        reward = 0
        
        for episode in range(EPISODE_SIZE):

            # Evaluate the state :
            stacked_observation = framestack.get_stacked_frames_tensor()

            # Calculate advantage estimation
            value_estimate = value_function.forward(stacked_observation)

            # Calculation actions probability
            action_probabilities = new_policy.forward(stacked_observation)
            old_action_probabilities = old_policy.forward(stacked_observation)

            # Chosing the action
            action = torch.argmax(action_probabilities, dim=1)
            if epoch < 55:
                if randint(1,100) >= 10:  # Epsilon greedy
                    action = torch.tensor([env.action_space.sample()], device=device)
            observation, reward, done, truncated, info = env.step(action)                

            if info.get("lives", 0) == 0:
                observation, info = env.reset()
                obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
                framestack.reset(obs)

            processed_obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
            framestack.add_frame(processed_obs)
            stacked_observation = framestack.get_stacked_frames_tensor()
            
            custom_reward = get_reward(info.get("lives",0))

            # RND Reward Calculation
            flattened_obs = torch.flatten(torch.tensor(rgbToGray(observation), dtype=torch.float32, device=device).unsqueeze(0))
            predicted_state = predictor_net(flattened_obs)
            target_state = target_net(flattened_obs)

            rnd_reward = torch.mean((predicted_state - target_state) ** 2) # MSE loss between the networks

            if epoch < 150:
                predictor_optimizer.zero_grad()
                loss = rnd_reward + predictor_net.l1_regularization()
                loss.backward()
                predictor_optimizer.step()
            else:
                predictor_optimizer_late.zero_grad()
                loss = rnd_reward + predictor_net.l1_regularization()
                loss.backward()
                predictor_optimizer_late.step()

            full_reward = torch.tensor([custom_reward + 20*reward + 10*rnd_reward.item()], requires_grad=False, dtype=torch.float32).unsqueeze(0)
            rewards_list.append([custom_reward, 20*reward, 10*rnd_reward.item()])

            # Save the episode values
            value_function_values.append(value_estimate)
            old_policy_values.append(old_action_probabilities)
            new_policy_values.append(action_probabilities)
            rewards.append(full_reward)

        discounted_rewards = calculate_discounted_rewards(rewards, gamma=0.99, future_rewards_count=4)
        discounted_rewards = torch.stack([torch.tensor(dr, dtype=torch.float32, device=device) for dr in discounted_rewards])

        old_policy_values = torch.stack(old_policy_values)
        new_policy_values = torch.stack(new_policy_values)  
        old_log_probs = torch.log(old_policy_values + 1e-8)
        new_log_probs = torch.log(new_policy_values + 1e-8)
        data_per_epochs.append([value_function_values, old_log_probs, new_log_probs, discounted_rewards])

        # Calculate both losses and backward
        value_function_values = torch.stack(value_function_values)
        value_function_loss = F.mse_loss(value_function_values, discounted_rewards) + value_function.l1_regularization()
        
        optimizer_value.zero_grad()
        value_function_loss.backward()
        optimizer_value.step()

        stacked_advantages = discounted_rewards - value_function_values.detach()
        ppo_loss = ppo_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=stacked_advantages
        )

        optimizer_policy.zero_grad()
        ppo_loss += new_policy.l1_regularization()
        ppo_loss.backward()
        optimizer_policy.step()

        list_loss.append(float(value_function_loss))

        if epoch == num_epochs-1 and params[2] != None:
            print("Saving policy network\n")
            torch.save(new_policy.state_dict(), params[2])
        
        if epoch == num_epochs-1 and params[3] != None:
            print("Saving value network\n")
            torch.save(value_function.state_dict(), params[3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=20)
    parser.add_argument('--policy_model', default=None)
    parser.add_argument('--value_model', default=None)
    parser.add_argument('--save_path_policy', default=None)
    parser.add_argument('--save_path_value', default=None)

    args = parser.parse_args()

    train(
        num_epochs=args.num_epochs,
        policy_model=args.policy_model,
        value_model=args.value_model,
        save_path_policy=args.save_path_policy,
        save_path_value=args.save_path_value,
    )