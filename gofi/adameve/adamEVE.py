import torch
from torch.optim import Optimizer
import math

class AdamEVE(Optimizer):
    r"""
    Adam-EVE optimizer: Adaptive Moment Estimation with Escape Velocity Enhancement.
    Extends Adam with a stochastic secondary momentum term to escape saddle points and shallow minima.

    b_n = a_n + α_n * ã_{n-1}
    where ã_{n-1} ~ N(a_{n-1}, σ^2), and α_n = max(1, loss) * max(1, f(||grad||))

    Arguments:
        params (iterable): model parameters
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): Adam coefficients (default: (0.9, 0.999))
        eps (float, optional): term added to denominator (default: 1e-8)
        gamma (float, optional): scaling for perturbation std (default: 0.1)
        f_max (float, optional): scaling factor for gradient-based α (default: 10.0)
        p (float, optional): power for gradient norm in f(||grad||) (default: 1.0)
        weight_decay (float, optional): weight decay (default: 0)
        adam_kwargs (dict): extra args passed to internal Adam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 gamma=0.1, f_max=10.0, p=1.0, weight_decay=0.0):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        gamma=gamma, f_max=f_max, p=p,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        # internal adam state
        self.adam = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.prev_update = [torch.zeros_like(p.data) for p in self.adam.param_groups[0]['params']]

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        if closure is not None:
            loss = closure()

        loss_value = loss.item() if loss is not None else 1.0

        grad_norm = 0.0
        for group in self.adam.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(grad_norm) + 1e-12

        # f(||grad||) = f_max / (1 + grad_norm^p)
        for group in self.adam.param_groups:
            gamma = group['gamma']
            f_max = group['f_max']
            p = group['p']
            alpha = max(1.0, loss_value) * max(1.0, f_max / (1.0 + grad_norm**p))

            self.adam.step()  # perform standard Adam update first

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                a_n = -group['lr'] * p.grad.data.clone()

                # stochastic perturbation around previous update
                noise_std = gamma * a_n.abs()
                a_tilde_prev = self.prev_update[i] + torch.randn_like(a_n) * noise_std

                b_n = a_n + alpha * a_tilde_prev

                p.data.add_(b_n)  # perform modified update
                self.prev_update[i] = b_n.clone()

        return loss
