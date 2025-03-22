import torch

def ppo_policy_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    # Calcul du ratio r_t(theta)
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Premier terme : ratio * avantage
    surr1 = ratio * advantages
    
    # Second terme : clip le ratio dans [1 - epsilon, 1 + epsilon]
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    # On prend le minimum des deux termes
    loss = -torch.min(surr1, surr2).mean()  # Moyenne empirique (approximation de l'espérance)
    
    return loss