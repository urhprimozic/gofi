import torch
from gofi.graphs.recursive.model import PermutationGenerator

def loss_qap(P, M1, M2):
    return torch.norm(M1 - P @ M2 @ P.T)

def loss_qap_normalized(P, M1, M2):
    n = P.shape[0]
    return torch.norm(M1 - P @ M2 @ P.T) / (n**2)

def reinforce_loss(generator, M1, M2, baseline, batch_size=100, baseline_decay=1, moving_average=True):
    P, log_probs = generator(batch_size=batch_size)

    diff = M1.unsqueeze(0) - P @ M2.unsqueeze(0) @ P.transpose(1, 2)
    losses = torch.norm(diff, dim=(1, 2)) ** 2

    # baseline update
    with torch.no_grad():
        if moving_average:
            batch_mean = losses.mean()
            baseline.mul_(baseline_decay).add_((1 - baseline_decay) * batch_mean)
        else: 
            baseline =  losses.mean()

    advantages = losses - baseline
    
    loss = (advantages.detach() * (log_probs)).mean()

    return loss, losses.mean().item()



def sample_losses_and_perms(generator, M1, M2, batch_size=32):
    """
    Returns matrices, probabilities, losses
    """
    matrices, log_probs = generator(batch_size=batch_size)   # (batch_size, n)

    losses = []
    for i in range(batch_size):
        P = matrices[i]
        losses.append(loss_qap(P, M1, M2))
    
    return matrices, torch.exp(log_probs), losses