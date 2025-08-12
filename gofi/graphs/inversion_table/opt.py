from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.loss import norm_loss_normalized
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch 


def training(
    dist: PermModel,
    M1: torch.Tensor,
    M2: torch.Tensor,
    loss_function=norm_loss_normalized,
    eps=None,
    max_steps=None,
    adam_parameters=None,
    scheduler = None,
    scheduler_parameters=None,
    scheduler_input=None,
    grad_clipping=100,
    verbose=500,
):
    """
    Trains a PermModel dist on graphs, given with adjecency matrices M1 and M2. 

    Parameters
    ----------
    dist : PermModel
        model of the distribution on S_n
    
    eps : 
    
    """
    if eps is None and max_steps is None:
        raise ValueError("Both eps and max_spets cannot be None. At least one must be given.")
    # prepare optimiser
    if adam_parameters is None:
        adam_parameters = {}
    opt = Adam(dist.model.parameters(), **adam_parameters)

    if scheduler is not None:
        scheduler = scheduler(opt, **scheduler_parameters)

    step = 0
    
    while True:
        step += 1

        #################### one step of opt ####################
        loss = loss_function(dist, M1, M2)
        loss.backward()

        if grad_clipping is not None:
            clip_grad_norm_(dist.model.parameters(), max_norm=grad_clipping)

        opt.step()
        # reset for next iter
        opt.zero_grad()
        # clear cahce
        dist.clear_cache()

        ### log
        if verbose and ( (step % verbose == 1) or verbose == 1):
            print(f"step {step} loss: {loss.item()}")

        #################### stopping conditions ####################
        if max_steps is not None:
                if step >= max_steps:
                    if verbose:
                        print(f"Stopping after {step} steps. Last loss: {loss.item()}")
                    break
        
        if eps is not None:
            if loss < eps:
                if verbose:
                    print(f" Loss of {loss.item()} < {eps} reached. Stopping training.")
                break
        #################### scheduler ####################
        if scheduler is not None:
            if scheduler_input is not None:
                if scheduler_input == 'loss':
                    scheduler.step(loss)
            else:
                scheduler.step()
    
    
