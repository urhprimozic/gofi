from models import GeneratorModel
from typing import Callable, Any
import torch
from tqdm import tqdm

def training_loop(
    model: GeneratorModel,
    optimizer,
    n_steps: int,
    loss_function: Callable[[GeneratorModel], Any],
    eps=1e-3,
    verbose=False,
    extract_loss_info: Callable[[GeneratorModel], Any] | None = None,
    scheduler : torch.optim.lr_scheduler.LRScheduler | None = None,
    loading_bar : bool =False
):
    """
    Optimisation of model parameters to minimize loss.

    Parameters
    -----------
    model : GeneratorModel
        GeneratorModel, which models a group representation
    optimizer
        torch optimizer for gd
    n_steps : int
        Number of steps in training loop.
    loss_function : Callable[[GeneratorModel],Any]
        Function that computes loss of the model.
    eps : int
        Loss threshold. If loss < eps, optimisation stops.
    verbose : boolean
        If true, there is logging
    extract_loss_info : Callable[[GeneratorModel], Any] | None
        If not none, every step extract_loss_info(model) is called and stored into table info.
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        If not none, lr scheduler is used while training.
    loading_bar : bool
        If true, tqdm loading bar is used to count the steps

    Returns
    ----------
    If extract_loss_info is None

    (loss, converged) : tuple (list, bool)
        where 
    loss : list
        List of losses.
    converged: bool
        True if training converged
    
    Else

    (loss, info, converged) : tuple[list, list, bool]
        where

    loss : list
        List of losses.
    info : list
        List of infos of every step of optimisation.
    converged: bool
        True if training converged

    """

    losses = []
    if extract_loss_info is not None:
        infos = []

    iterator = range(n_steps)
    if loading_bar:
        iterator = tqdm(iterator, total=n_steps)
    
    for i in iterator:
        # calculate loss
        loss = loss_function(model)

        # save infos
        if extract_loss_info is not None:
            infos.append(extract_loss_info(model))

        # save current loss value
        losses.append(loss.item())

        # gradient descent
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # lr scheduler
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(loss)
            else:
                scheduler.step()

        # check for convergence
        if loss < eps:
            if verbose:
                print(f"Converged at step {i}")
                if extract_loss_info is not None:
                    return losses, infos, True
                return losses, True
    if verbose:
        print(f"Failed to converge after {n_steps} steps. Last loss = {loss.item()}")
    if extract_loss_info is not None:
        return losses, infos, False
    return losses, False
