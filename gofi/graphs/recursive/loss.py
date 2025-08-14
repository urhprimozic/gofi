import torch

def loss_qap(P, M1, M2):
    return torch.norm(M1 - P @ M2 @ P.T)

def loss_qap_normalized(P, M1, M2):
    n = P.shape[0]
    return torch.norm(M1 - P @ M2 @ P.T) / (n**2)

def reinforce_loss(generator, M1, M2, baseline, batch_size=16, baseline_decay=0.9):
    P, log_probs = generator(batch_size=batch_size)

    diff = M1.unsqueeze(0) - P @ M2.unsqueeze(0) @ P.transpose(1, 2)
    losses = torch.norm(diff, dim=(1, 2)) ** 2

    # baseline update
    with torch.no_grad():
        batch_mean = losses.mean()
        baseline.mul_(baseline_decay).add_((1 - baseline_decay) * batch_mean)

    advantages = losses - baseline
    loss = -(advantages.detach() * (-log_probs)).mean()

    return loss, losses.mean().item()


