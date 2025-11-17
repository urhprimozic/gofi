import torch
from torch.optim import Adam
import math

class AdamWithNoise(Adam):
    """
    Adam optimizer + additive Gaussian perturbation when gradient norm is small.
    Idea: follow Jin et al. style perturbed gradient descent — when ||grad|| < threshold,
    add a small Gaussian perturbation to parameters to escape saddle points.

    Usage:
        opt = AdamPerturbed(model.parameters(), lr=1e-3,
                            noise_scale=1e-5, grad_threshold=1e-3,
                            cooldown_steps=10, decay=0.99)

        for x, y in loader:
            opt.zero_grad()
            loss = model_loss(x, y)
            loss.backward()
            opt.step()
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0,
                 # perturbation-specific
                 noise_scale=1e-5,
                 grad_threshold=1e-3,
                 cooldown_steps=10,
                 decay=1.0,
                 noise_max=None,
                 sample_on_gpu=True,
                 verbose=False,
                 **adam_kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, **adam_kwargs)
        self.noise_scale = float(noise_scale)
        self._base_noise_scale = float(noise_scale)
        self.grad_threshold = float(grad_threshold)
        self.cooldown_steps = int(cooldown_steps)
        self.decay = float(decay)  # multiply noise_scale by decay after each perturb
        self.noise_max = None if noise_max is None else float(noise_max)
        self.sample_on_gpu = bool(sample_on_gpu)
        self.verbose = bool(verbose)

        # per-parameter cooldown counters
        self._cooldowns = {}
        # global step counter
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform Adam step, then maybe add perturbation."""
        loss = None
        if closure is not None:
            loss = closure()

        # First do the normal Adam update
        super().step(closure=closure)
        self._step_count += 1

        # compute global gradient L2 norm (across all params)
        total_norm_sq = 0.0
        device = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                device = g.device if device is None else device
                total_norm_sq += float(g.norm(2).item() ** 2)
        g_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0

        # Decide whether to perturb: global check + per-param cooldown
        do_perturb = (g_norm < self.grad_threshold)

        if do_perturb:
            if self.verbose:
                print(f"[AdamPerturbed] step {self._step_count}: grad_norm={g_norm:.3e} < {self.grad_threshold:.3e} -> perturbing")
        else:
            # decrement cooldowns if any
            for k in list(self._cooldowns.keys()):
                if self._cooldowns[k] > 0:
                    self._cooldowns[k] -= 1
                    if self._cooldowns[k] <= 0:
                        del self._cooldowns[k]
            return loss

        # apply perturbation only to parameters that are not in cooldown
        for group in self.param_groups:
            lr = group.get('lr', 1e-3)
            for p in group['params']:
                if p.grad is None:
                    continue
                key = id(p)
                if self._cooldowns.get(key, 0) > 0:
                    # skip this param (still cooling down)
                    continue

                # compute parameter-wise noise std.
                # default: isotropic small noise scaled by noise_scale
                # optionally scale with lr and parameter magnitude to be relative
                param_scale = float(p.data.abs().mean().item() + 1e-12)
                noise_std = self.noise_scale * max(lr, param_scale)

                if self.noise_max is not None:
                    if noise_std > self.noise_max:
                        noise_std = self.noise_max

                # sample noise (on same device/dtype)
                if self.sample_on_gpu and p.data.is_cuda:
                    noise = torch.randn_like(p.data, device=p.data.device) * noise_std
                else:
                    noise = torch.randn_like(p.data) * noise_std

                # add perturbation to parameters
                p.data.add_(noise)

                # set cooldown for this param
                if self.cooldown_steps > 0:
                    self._cooldowns[key] = self.cooldown_steps

        # optionally decay the noise scale
        if self.decay != 1.0:
            self.noise_scale *= self.decay
            # keep a floor so it doesn't vanish to zero accidentally
            self.noise_scale = max(self.noise_scale, 1e-12)

        return loss
