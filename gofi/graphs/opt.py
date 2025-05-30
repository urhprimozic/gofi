import torch
from torchdiffeq import odeint, odeint_event
from gofi.models import RandomMap
from typing import Callable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from gofi.graphs.loss import (
    BijectiveLoss,
    RelationLoss,
    BijectiveLossMatrix,
    RelationLossMatrix,
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.nn as nn


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
):
    def loss_function(f, M1, M2):
        return A * BijectiveLoss(f) + B * RelationLoss(f, M1, M2)

    if adam_parameters is None:
        adam_parameters = {}
    opt = Adam(f.parameters(), **adam_parameters)

    if scheduler is not None:
        scheduler = scheduler(opt, **scheduler_parameters)


    grad_norm = eps + 1
    step = 0
    # train till norm of gradient is small - local minima
    try:
        while grad_norm > eps:
            step += 1
            if verbose and ( (step % verbose == 1) or verbose == 1):
                print(f"Step {step}", end=". ")
            # one step of training
            
            loss = loss_function(f, M1, M2)
            loss.backward()
            # clip gradient
            if grad_clipping is not None:
                clip_grad_norm_(f.parameters(), max_norm=grad_clipping)

            opt.step()
            


            # update grad norm
            grad_norm = 0.0
            for p in f.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # L2 norm
                    grad_norm += param_norm.item() ** 2
            if verbose and ( (step % verbose == 1) or verbose == 1):
                print(f"Loss= {loss.item()}, |grad| = {grad_norm}", end="")
                if scheduler is not None:
                        try:
                            print(f", lr={scheduler.get_last_lr()}", end="")
                        except: 
                            pass
            print("")
            # stop when reaching max steps
            if max_steps is not None:
                if step >= max_steps:
                    break
            
            # reset for next iter
            opt.zero_grad()
            
            if scheduler is not None:
                if scheduler_input is not None:
                    if scheduler_input == 'loss':
                        scheduler.step(loss)
                else:
                    scheduler.step()
    except:
        print("exception.. returning f")
        return f 


def solve_gradient_flow(phi0: torch.Tensor, M1: torch.Tensor, M2: torch.Tensor):
    def neg_grad(t, x):
        x.requires_grad_ = True
        loss = RelationLossMatrix(torch.softmax(x, dim=1), M1, M2)
        grad = torch.autograd.grad(loss, x)[0]  # , retain_graph=True)
        return grad

    return odeint(neg_grad, phi0, torch.linspace(0, 60, 6000))
