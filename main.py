from visualisation.visualize import visualize_state, visualize_frame_stack
from preprocessing.preprocess import preprocess_observation
from config import WIDTH, HEIGHT
from frame_stack import FrameStack
from models.value_function_model import ValueFunction
from models.policy_model import PolicyModel

import numpy as np
import gymnasium as gym
import ale_py

def main():

    # Create the environment
    render_mode = "human"
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
    
    # For observation resizing
    downscale = 0.9
    new_height, new_width = (int(HEIGHT*downscale),int(WIDTH*downscale))

    framestack = FrameStack()
    value_function = ValueFunction()
    policy = PolicyModel()

    # Reset the environment to get the initial state
    observation, info = env.reset()
    obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
    framestack.reset(obs)

    done = False
    i = 0
    # Take an action and get the next state
    while not done:
        i += 1
        action = env.action_space.sample()  # Sample a random action (replace this with your policy/action)
        observation, reward, done, truncated, info = env.step(action)
        processed_obs = preprocess_observation(observation, new_width=new_width, new_height=new_height)
        if i%2:
            framestack.add_frame(processed_obs)
        if i == 10:
            done=True

    """
    visualize_state(processed_obs)
    
    """
    visualize_frame_stack(framestack)
    print(value_function.forward(framestack.get_stacked_frames_tensor()))
    print(policy.forward(framestack.get_stacked_frames_tensor()))

if __name__ == "__main__":
    main()