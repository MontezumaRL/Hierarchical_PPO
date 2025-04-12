import numpy as np
import torch
from collections import deque

class FrameStack:

    def __init__(self, stack_size=3):
        self.stack_size = stack_size
        self.frames = deque(maxlen=self.stack_size)  # Deque with a fixed size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self, initial_frame):
        # Initialize deque with the initial frame repeated for the stack size
        self.frames = deque([initial_frame] * self.stack_size, maxlen=self.stack_size)

    def add_frame(self, new_frame):
        self.frames.append(new_frame)  # Add the new frame to the deque
        return np.stack(list(self.frames), axis=0)

    def get_stacked_frames_tensor(self):
        return torch.tensor(np.stack(list(self.frames), axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def get_stacked_frames(self):
        return np.stack(list(self.frames), axis=0)