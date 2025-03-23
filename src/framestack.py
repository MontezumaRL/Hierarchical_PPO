


from collections import deque

import numpy as np

from src.utils import preprocess_frame


class FrameStack:
    
    def __init__(self, env, k=4):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        obs = self.env.reset()
        processed_obs = preprocess_frame(obs[0])
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return np.array(self.frames).reshape(4, 84, 84)

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        processed_obs = preprocess_frame(obs)
        self.frames.append(processed_obs)
        return np.array(self.frames).reshape(4, 84, 84), reward, done, info