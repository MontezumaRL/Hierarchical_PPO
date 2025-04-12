import torch
from config import CLIP_EPSILON

def ppo_policy_loss(old_log_probs, new_log_probs, advantages, epsilon=CLIP_EPSILON):
    # probabilities ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    # Minimum between clipping and without
    loss = -torch.min(surr1, surr2).mean()
    
    return loss