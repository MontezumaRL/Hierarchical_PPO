from config import EPISODE_SIZE, CLIP_EPSILON, WIDTH, HEIGHT
from models.policy_model import PolicyModel
from models.value_function_model import ValueFunction
from preprocessing.preprocess import preprocess_observation, rgbToGray
from frame_stack import FrameStack
from reward import get_reward, calculate_discounted_rewards
import torch
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
from random import randint
from loss import ppo_policy_loss
from rnd import PredictorNet, TargetNet

def goback(env):
    for i in range(19):
        env.step(0)
    env.step(11)
    for i in range(20):
        env.step(0)
        #print(done)
    env.step(11)
    for i in range(5):
        env.step(0)
    env.step(3)
    for i in range(5):
        env.step(0)
    env.step(3)
    for j in range(12):
        for i in range(5):
            env.step(0)
        env.step(5)
    for j in range(8):
        for i in range(5):
            env.step(0)
        env.step(4)
    env.step(12)
    for j in range(15):
        for i in range(3):
            env.step(0)
        env.step(4)
    for j in range(7):
        for i in range(4):
            env.step(0)
        env.step(4)
    for j in range(7):
        for i in range(4):
            env.step(0)
        env.step(2)

def change_pos(x,y,env, framestack, new_width, new_height):
    env.unwrapped.ale.setRAM(42,x)
    env.unwrapped.ale.setRAM(43,y)
    for i in range(8):
        observation, reward, done, truncated, info = env.step(0)
    obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
    framestack.reset(obs)
    
    
def train(num_epochs=2000):

    # Init of necessary objects
    old_policy = PolicyModel()
    new_policy = PolicyModel()
    new_policy.copy_weights_from(old_policy)
    value_function = ValueFunction()
    predictor_net = PredictorNet(input_dim=WIDTH * HEIGHT)  # Assuming flattened image input
    target_net = TargetNet(input_dim=WIDTH * HEIGHT)

    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")
    framestack = FrameStack()
    optimizer_policy = torch.optim.Adam(new_policy.parameters(), lr=0.001)
    optimizer_value = torch.optim.Adam(value_function.parameters(), lr=0.003)
    predictor_optimizer = torch.optim.Adam(predictor_net.parameters(), lr=0.002)

    for param in target_net.parameters():
        param.requires_grad = False
    # For observation resizing
    downscale = 0.9
    new_height, new_width = (int(HEIGHT*downscale),int(WIDTH*downscale))

    observation, info = env.reset()

    change_pos(20,195,env, framestack, new_width, new_height)

    data_per_epochs = []
    list_loss = []
    nb_lifes = 5
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

            # Calculate rewards and advantage estimation

            value_estimate = value_function.forward(stacked_observation)
            #print(f"pred : {value_estimate}, reward : {full_reward}")
            # Calculation actions probability
            action_probabilities = new_policy.forward(stacked_observation)
            old_action_probabilities = old_policy.forward(stacked_observation)

            #   Chosing the action
            action = torch.argmax(action_probabilities, dim=1)
            if epoch < 35:
                if randint(1,100) >= 5:  # Random move possibility
                    action = torch.tensor([env.action_space.sample()])
            observation, reward, done, truncated, info = env.step(action)                
            if reward != 0:
                print("\n\n\n-----------------------------------REWARD TAKEN-----------------------------------\n\n\n")

            if info.get("lives", 0) == 0:
                env.reset()
                nb_lifes = 5
                change_pos(20,195,env, framestack, new_width, new_height)

            if info.get("lives", 0) != nb_lifes:
                for i in range(20):
                    env.step(0)
                change_pos(30,160,env, framestack, new_width, new_height)
            nb_lifes = info.get("lives", 0)



            processed_obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
            framestack.add_frame(processed_obs)
            
            obs = env.unwrapped.ale.getRAM()
            x_pos = obs[42]
            y_pos = obs[43]
            #print(f"Position du personnage : x = {x_pos}, y = {y_pos}")
            custom_reward = get_reward(stacked_observation, info.get("lives",0), x_pos, y_pos)
            last_action = action

            # RND Reward Calculation
            flattened_obs = torch.flatten(torch.tensor(rgbToGray(observation), dtype=torch.float32).unsqueeze(0))
            predicted_state = predictor_net(flattened_obs)
            target_state = target_net(flattened_obs)
            rnd_reward = torch.mean((predicted_state - target_state) ** 2) # MSE loss between the networks
            predictor_optimizer.zero_grad()
            loss = rnd_reward + predictor_net.l1_regularization()
            loss.backward()
            predictor_optimizer.step()
            #print(rnd_reward.item())

            full_reward = torch.tensor([custom_reward + 10*reward + rnd_reward.item()], requires_grad=False, dtype=torch.float32).unsqueeze(0)
            # Récupérer les données de l'episode
            value_function_values.append(value_estimate)
            old_policy_values.append(old_action_probabilities)
            new_policy_values.append(action_probabilities)
            rewards.append(full_reward)

        discounted_rewards = calculate_discounted_rewards(rewards, gamma=0.99, future_rewards_count=4)
        discounted_rewards = torch.stack([torch.tensor(dr, dtype=torch.float32) for dr in discounted_rewards])
        
        #print(f"discounted rewards : {discounted_rewards}\n")

        old_policy_values = torch.stack(old_policy_values)  # [EPISODE_SIZE, num_actions]
        new_policy_values = torch.stack(new_policy_values)  
        old_log_probs = torch.log(old_policy_values + 1e-8)
        new_log_probs = torch.log(new_policy_values + 1e-8)
        data_per_epochs.append([value_function_values, old_log_probs, new_log_probs, discounted_rewards])

        # Calculate both losses
        value_function_values = torch.stack(value_function_values)
        value_function_loss = F.mse_loss(value_function_values, discounted_rewards) + value_function.l1_regularization()
        
        optimizer_value.zero_grad()
        value_function_loss.backward()
        optimizer_value.step()

        stacked_advantages = discounted_rewards - value_function_values.detach()
        #print(stacked_advantages)
        ppo_loss = ppo_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=stacked_advantages
        )

        optimizer_policy.zero_grad()
        ppo_loss += new_policy.l1_regularization()
        ppo_loss.backward()
        optimizer_policy.step()

        print(f"loss : {ppo_loss} et value loss : {torch.mean(stacked_advantages)}")
        list_loss.append(float(value_function_loss))
        #print(list_loss)
        if epoch == 20:
            rewards = [sublist[3] for sublist in data_per_epochs]
            print(rewards)
            #plt.plot(data_per_epochs[:][4])
            #plt.show()

train()