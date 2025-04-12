import gymnasium
from collections import deque
import numpy as np
import torch
from preprocessing.preprocess import preprocess_frame
import ale_py
import matplotlib.pyplot as plt

class MontezumaEnvironment:
    def __init__(self, render_mode = "rgb_array"):
        self.env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
        self.frame_stack = deque(maxlen=4)
        self.action_space = self.env.action_space
        self.n_actions = self.env.action_space.n
        self.lives = 0

    def reset(self):
        state = self.env.reset()[0]
        state = preprocess_frame(state)

        # Initialise framestack
        for _ in range(4):
            self.frame_stack.append(state)
        
        # Initialise lifes
        _, _, _, _, info = self.env.step(0)  # NOOP
        self.lives = info.get('lives', 0)

        return np.array(self.frame_stack)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = preprocess_frame(next_state)
        self.frame_stack.append(next_state)

        done = terminated or truncated

        # Vérifier si une vie a été perdue
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives
        self.lives = current_lives

        if life_lost:
            reward -= 10.0
            done = True

        return np.array(self.frame_stack), reward, done, info

    def get_state_tensor(self, device):
        return torch.FloatTensor(np.array(self.frame_stack)).unsqueeze(0).to(device)

    def display_frame_stack(self, save_path="frame_stack.png"):
        """Affiche les 4 frames de la stack actuelle ou les sauvegarde dans un fichier"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, frame in enumerate(self.frame_stack):
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f'Frame {i+1}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Frame stack sauvegardée dans {save_path}")

    def close(self):
        self.env.close()