from torch.autograd.functional import jacobian
from torchdiffeq import odeint, odeint_event
from gofi.models import RandomMap
from typing import Callable, Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from gofi.graphs.loss import (
    BijectiveLoss,
    RelationLoss,
    BijectiveLossMatrix,
    RelationLossMatrix, LossGraphMatching, LossGraphMatchingRandomMap
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch
import traceback

def training(
    f: RandomMap,
    M1: torch.Tensor,
    M2: torch.Tensor,
    eps=0.001,
    max_steps=None,
    adam_parameters=None,
    scheduler = None,
    scheduler_parameters=None,
    scheduler_input=None,
    grad_clipping=10,
    verbose=500,
    A=1,
    B=1,
    use_relation_loss=True,
    store_relation_loss=False
):
    if use_relation_loss:
        def loss_function(f, M1, M2):
            return A * BijectiveLoss(f) + B * RelationLoss(f, M1, M2) 
    else:
        def loss_function(f, M1, M2):
            return A * BijectiveLoss(f) + B * LossGraphMatchingRandomMap(f, M1, M2)

    if adam_parameters is None:
        adam_parameters = {}
    opt = Adam(f.parameters(), **adam_parameters)

    if scheduler is not None:
        scheduler = scheduler(opt, **scheduler_parameters)


    grad_norm = eps + 1
    step = 0
    losses = []
    relation_losses = []
    # train till norm of gradient is small - local minima
    try:
        #while grad_norm > eps:
        while True:
            opt.zero_grad()
            step += 1
            
            # one step of training
            
            loss = loss_function(f, M1, M2)
           

            loss.backward()
            # diagnostic about grads
            any_grad = False
            total_sq = 0.0
            for name, p in [(n, p) for n, p in f.named_parameters()]:
                g = p.grad
                if g is None:
                    print(f"  PARAM: {name}: grad is None (shape {tuple(p.shape)})")
                else:
                    any_grad = True
                    # avoid .data usage; detach the grad
                    norm = g.detach().norm().item()
                    total_sq += norm ** 2

            grad_norm = total_sq ** 0.5
            



            # clip gradient
            if grad_clipping is not None:
                clip_grad_norm_(f.parameters(), max_norm=grad_clipping)

            opt.step()

            with torch.no_grad():
                loss_value = loss.item()
                losses.append(loss_value)
                if store_relation_loss:
                    p_matrix = f.mode_matrix()
                    rloss = LossGraphMatching(p_matrix, M1, M2)
                    relation_losses.append(rloss.item())
            


            
            if verbose and ( (step % verbose == 1) or verbose == 1):
                print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e} ||gradient||={grad_norm:.6f}", end="\r")
            # stop when reaching max steps
            if max_steps is not None:
                if step >= max_steps:
                    break
            

            
            if scheduler is not None:
                if scheduler_input is not None:
                    if scheduler_input == 'loss':
                        scheduler.step(loss)
                else:
                    scheduler.step()
            

    except Exception as e:
        print(f"Exception: {e}")
        print("--------------------------\nTre::\n")
        traceback.print_exc()
        print("--------------------------\nException occured..")

    if verbose:
        print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e} ||gradient||={grad_norm:.6f}")
    if store_relation_loss:
        return losses, relation_losses
    return losses


def training_stable(
    f: RandomMap,
    M1: torch.Tensor,
    M2: torch.Tensor,
    eps=0.001,
    max_steps=None,
    adam_parameters=None,
    scheduler = None,
    scheduler_parameters=None,
    scheduler_input=None,
    grad_clipping=10,
    verbose=500,
    A=1,
    B=1,
    use_relation_loss=True,
    store_relation_loss=False,
    # anti-vanish knobs (opt-in; keep default False/0 to preserve old behavior)
    anti_vanish: bool = False,
    warmup_steps: int = 0,
    entropy_weight: float = 0.0,
    entropy_decay: float = 1.0,
    grad_noise_std: float = 0.0,
    sinkhorn_warmup_disable: bool = False,
    sinkhorn_iters_warmup: Optional[int] = None,
    sinkhorn_iters_post: Optional[int] = None,
    # normalize graph-matching loss (only used when use_relation_loss=False)
    normalize_graph_matching: bool = False,
):
    if use_relation_loss:
        def loss_function(f, M1, M2):
            return A * BijectiveLoss(f) + B * RelationLoss(f, M1, M2) 
    else:
        if normalize_graph_matching:
            def loss_function(f, M1, M2):
                n = M1.shape[0]
                return A * BijectiveLoss(f) + B * (LossGraphMatchingRandomMap(f, M1, M2) / (n ** 2))
        else:
            def loss_function(f, M1, M2):
                return A * BijectiveLoss(f) + B * LossGraphMatchingRandomMap(f, M1, M2)

    if adam_parameters is None:
        adam_parameters = {}
    opt = Adam(f.parameters(), **adam_parameters)

    if scheduler is not None:
        scheduler = scheduler(opt, **scheduler_parameters)

    # cache original sinkhorn config to restore after warmup
    _orig_sinkhorn = getattr(f, "sinkhorn", None)
    _orig_sinkhorn_iters = getattr(f, "sinkhorn_iters", None)
    current_entropy_weight = float(entropy_weight)

    grad_norm = eps + 1
    step = 0
    losses = []
    relation_losses = []
    try:
        while grad_norm > eps:
            opt.zero_grad()
            step += 1

            # Optional warmup scheduling (row-softmax early, Sinkhorn later)
            if anti_vanish and hasattr(f, "sinkhorn"):
                if step <= warmup_steps:
                    if sinkhorn_warmup_disable:
                        f.sinkhorn = False
                    if sinkhorn_iters_warmup is not None:
                        f.sinkhorn_iters = sinkhorn_iters_warmup
                else:
                    if _orig_sinkhorn is not None:
                        f.sinkhorn = _orig_sinkhorn
                    if sinkhorn_iters_post is not None:
                        f.sinkhorn_iters = sinkhorn_iters_post
                    elif _orig_sinkhorn_iters is not None:
                        f.sinkhorn_iters = _orig_sinkhorn_iters

            base_loss = loss_function(f, M1, M2)

            # Max-entropy regularization to keep gradients alive (annealed)
            if anti_vanish and current_entropy_weight != 0.0:
                P = f.P()
                Pl = P.clamp_min(1e-9)
                entropy = -(Pl * Pl.log()).sum() / P.numel()
                base_loss = base_loss - current_entropy_weight * entropy
                current_entropy_weight *= float(entropy_decay)

            loss = base_loss
            loss.backward()

            # optional gradient noise injection
            total_sq = 0.0
            for _, p in f.named_parameters():
                if p.grad is not None:
                    if anti_vanish and grad_noise_std > 0.0:
                        p.grad.add_(torch.randn_like(p.grad) * grad_noise_std)
                    total_sq += float(p.grad.detach().norm().item() ** 2)
            grad_norm = total_sq ** 0.5

            if grad_clipping is not None:
                clip_grad_norm_(f.parameters(), max_norm=grad_clipping)

            opt.step()

            with torch.no_grad():
                loss_value = float(loss.item())
                losses.append(loss_value)
                if store_relation_loss:
                    p_matrix = f.mode_matrix()
                    rloss = LossGraphMatching(p_matrix, M1, M2)
                    relation_losses.append(float(rloss.item()))

            if verbose and ((step % verbose == 1) or verbose == 1):
                print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e} ||gradient||={grad_norm:.6f}", end="\r")

            if max_steps is not None and step >= max_steps:
                break

            if scheduler is not None:
                if scheduler_input == 'loss':
                    scheduler.step(loss)
                else:
                    scheduler.step()

    except Exception as e:
        print(f"Exception: {e}")
        print("--------------------------\nTre::\n")
        traceback.print_exc()
        print("--------------------------\nException occured..")

    if verbose:
        print(f"[step {step}] loss={loss_value:.6f} lr={opt.param_groups[0]['lr']:.2e} ||gradient||={grad_norm:.6f}")
    if store_relation_loss:
        return losses, relation_losses
    return losses

def integrate_gradient_flow(phi0: torch.Tensor, M1: torch.Tensor, M2: torch.Tensor, t_max : int = 10, steps : int = 1000):
    def neg_grad(t, x):
        x.requires_grad_ = True
        loss = RelationLossMatrix(torch.softmax(x, dim=1), M1, M2) +  BijectiveLossMatrix(torch.softmax(x, dim=1))
        grad = torch.autograd.grad(loss, x)[0]  # , retain_graph=True)
        return -grad

    return odeint(neg_grad, phi0, torch.linspace(0, t_max, steps))

def solve_gradient_flow(phi0: torch.Tensor, M1: torch.Tensor, M2: torch.Tensor, eps=0.01):
    """
    Integrate the gradient flow ODE using odeint_event, stopping when the gradient norm < eps.
    """
    def neg_grad(t, x):
        x.requires_grad_=True
        loss = RelationLossMatrix(torch.softmax(x, dim=1), M1, M2) + BijectiveLossMatrix(torch.softmax(x, dim=1))
        grad = torch.autograd.grad(loss, x)[0]
        return -grad
    
    loss  =  lambda x : RelationLossMatrix(torch.softmax(x, dim=1), M1, M2 ) + BijectiveLossMatrix(torch.softmax(x, dim=1))

    def event(t, x):
        x = x.detach().clone()
        grad = jacobian(loss, phi0)
        grad_norm = torch.norm(grad)
        return grad_norm - eps

    t0 = torch.tensor(0.0)
  
    solution, _ = odeint_event(
        neg_grad,
        phi0,
        t0,
        event_fn=event
    )
    return solution
