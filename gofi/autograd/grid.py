from gofi.autograd.models import GeneratorModel
from gofi.autograd.torch_settings import device
from datetime import datetime
from gofi.groups import demo_S3
import itertools
import torch
import numpy as np
from gofi.autograd.training import training_loop
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Callable, Any

def generate_complex_grid(
    model: GeneratorModel,
    loss_function: Callable[[GeneratorModel], Any],
    min_param: float,
    max_param: float,
    grid_dim: int,
    opt_steps: int = 2000,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    run_name: str = None,
    eps: float = 1e-3,
    verbose: bool = False,
    save_on_each_step=False,
    plot_loss_on_each_step=False
):
    '''
    Runs a SGD/adam for each set of parameters.
    Each parameter is bounded by min and max. There are grid_dim steps between min and max for each parameter.
    The number of all steps taken is grid_dim^n_parameters = grid_dim^8

    Parameters
    ----------
    model : GeneratorModel
        Model of representation G -> GL(model.dim)
    loss_function :  Callable[[GeneratorModel], Any]
        Loss function, which calculates loss of a GeneratorModel model.
    min_param : float
        Minimal threshold of every initial parameter.
    max_param : float
        Maximal threshold of every initial parameter.
    grid_dim : int
        Dimensions of the generated grid.
    opt_steps=2000
        Maximal steps of optimisation used in training for each set of initial parameters.
    weight_decay : float (default=1e-4)
        Weight decay for adam optimiser

    lr: float (default = 0.001)
        Learning rate for optimisation
    run_name : str=None
        Name of the run. Used to label results
    eps : float (default=1e-3)
        Threshold for loss. If loss is lover than that, training stops.
    verbose : bool (default=False)
        If true, logs are printed
    save_on_each_step : bool (default=False)
        If true, losses and weights are saved on each step. Deprecated... 
    plot_loss_on_each_step : bool (default=False)
        Saves plots of loss for each set of parameters
    
    Returns
    -------------
    run_name, min_param, max_param, grid_dim : tuple
        where

    run_name : str
        name of the run
    '''
    pass

def param_grid(min_param: float, max_param: float, n_points: int, n_params: int):
    """
    Create a grid of parameters to optimize over.

    Returns
    -------
    Iterator over tuples, each tuple is a different set of initial parameters.
    """
    # generate grid of all different parameters
    individual_params = np.linspace(min_param, max_param, num=n_points).tolist()
    params = itertools.product(*[individual_params for _ in range(n_params)])
    return params


def param_list_to_tensor(initial_params_list: list, params_shape: torch.Size):
    initial_params = torch.tensor(initial_params_list).to(device)
    initial_params = torch.reshape(initial_params, params_shape)
    return initial_params


def generate_grid(
    model: GeneratorModel,
    loss_function: Callable[[GeneratorModel], Any],
    min_param: float,
    max_param: float,
    grid_dim: int,
    opt_steps: int = 2000,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    run_name: str = None,
    eps: float = 1e-3,
    verbose: bool = False,
    save_on_each_step=False,
    plot_loss_on_each_step=False
):
    """
    TODO


    Runs a SGD/adam for each set of parameters.
    Each parameter is bounded by min and max. There are grid_dim steps between min and max for each parameter.
    The number of all steps taken is grid_dim^n_parameters = grid_dim^8

    Parameters
    ----------
    model : GeneratorModel
        Model of representation G -> GL(model.dim)
    loss_function :  Callable[[GeneratorModel], Any]
        Loss function, which calculates loss of a GeneratorModel model.
    min_param : float
        Minimal threshold of every initial parameter.
    max_param : float
        Maximal threshold of every initial parameter.
    grid_dim : int
        Dimensions of the generated grid.
    opt_steps=2000
        Maximal steps of optimisation used in training for each set of initial parameters.
    weight_decay : float (default=1e-4)
        Weight decay for adam optimiser

    lr: float (default = 0.001)
        Learning rate for optimisation
    run_name : str=None
        Name of the run. Used to label results
    eps : float (default=1e-3)
        Threshold for loss. If loss is lover than that, training stops.
    verbose : bool (default=False)
        If true, logs are printed
    save_on_each_step : bool (default=False)
        If true, losses and weights are saved on each step. Deprecated... 
    plot_loss_on_each_step : bool (default=False)
        Saves plots of loss for each set of parameters
    
    Returns
    -------------
    run_name, min_param, max_param, grid_dim : tuple
        where

    run_name : str
        name of the run
    
    TODO

    """
    # name the run - for saving results
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if verbose:
        print(f"Starting run {run_name}..")

    # collect info from model
    n_params = model.weights.numel()

    # generate grid of all different parameters
    params = param_grid(min_param, max_param, grid_dim, n_params)

    # losses
    all_losses = []
    all_conv_results = []
    # store weights
    all_weights = np.zeros([grid_dim**n_params] + list(model.params_shape))

    # run optimization for each set of parameters
    for index, initial_params_list in tqdm(enumerate(params), total=grid_dim**n_params):
        # get initial params
        initial_params = param_list_to_tensor(initial_params_list, model.params_shape)

        # create model with initial parameters
        model = model.get_new(initial_params)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(opt, "min")

        # run optimization of parameters
        losses, converged = training_loop(
            model=model,
            optimizer=opt,
            n_steps=opt_steps,
            loss_function=loss_function,
            eps=eps,
            verbose=verbose,
            scheduler=scheduler,
        )
        all_losses.append(losses)
        all_conv_results.append(converged)
        # save final weights
        final_weights = model.weights.cpu().detach().numpy()
        all_weights[index] = final_weights

        if plot_loss_on_each_step:
            # draw results
            plt.plot(losses, label="Loss")
            plt.title("Parameters: " + str(final_weights))
            plt.savefig(f"../../results/{run_name}_loss_plot_{index}.png")
            plt.clf()

        if save_on_each_step:    
            # save data
            # save final losses
            np.save(f"../../results/{run_name}_losses_{index}.npy", losses)
            
            np.save(f"../../results/{run_name}_weights_{index}.npy", final_weights)
    # save results 
    np.save(f"../../results/{run_name}_losses.npy", all_losses)
    np.save(f"../../results/{run_name}_weights.npy", all_weights)
    np.save(f"../../results/{run_name}_conv_results.npy", all_conv_results)

    return run_name, min_param, max_param, grid_dim

def plot_results(
    run_name: str,
    proj,
    min_param,
    max_param,
    grid_dimensions,
    model,
    eps=1e-3,
    title=None,
    x_label=None,
    y_label=None,
):
    """
    Plots 2D graph of speed of convergence/divergence of different parameters.
    x and y axis are defined as proj(model parameters) -> (x, y)

    Color of the point is defined by the speed, with which the model converged/diverged.

    Convergence speed = norm( #steps till loss < eps)
    Divergence speed =norm( if loss > eps: loss  )

    Parameters
    ----------
    run_name : str
        Name of the run, which data we want to plot.
    proj : function
        Function, which takes model parameters and returns 2D point
    """
    print("Warning: _print_results() is deprecated. Please use print_results() instead.")

    params_shape = model.params_shape
    n_params = n_params = model.weights.numel()

    params = param_grid(min_param, max_param, grid_dimensions, n_params)

    index_to_color_value = {}

    if title is None:
        title = f"Convergence speed of different initial parameters"

    # collect data
    print("Collecting data..")

    # load results 
    all_losses = np.load(f"../../results/{run_name}_losses.npy")
    all_weights = np.load(f"../../results/{run_name}_weights.npy")
    all_conv_results = np.load(f"../../results/{run_name}_conv_results.npy")

    raise NotImplementedError("refactor to new loading mechanism is notyet complete")

    for index in tqdm(range(grid_dimensions ** n_params), total=grid_dimensions**n_params):
        total_loss = np.load(f"../../results/{run_name}_losses_{index}.npy")
        # check if converges
        if (total_loss < eps).any():
            # get index of first step, where loss < eps
            color = np.where(total_loss < eps)[0][0].item()
            # normalise - faster closer to 1, slower is closer to 0
            color = 1 - color / total_loss.shape[0]
            index_to_color_value[index] = color
        else:
            # get last loss
            color = -total_loss[-1]
            index_to_color_value[index] = color

        # losses.append(np.load(f"../../results/{run_name}_losses_{index}.npy"))
        # weights.append(np.load(f"../../results/{run_name}_weights_{index}.npy"))

    # get axis

    # regenerate params
    params = param_grid(min_param, max_param, grid_dimensions, n_params)

    (x, y) = zip(
        *[
            proj(param_list_to_tensor(param_list, params_shape))
            for param_list in params
        ]
    )

    # plot data
    plt.scatter(
        x,
        y,
        c=[index_to_color_value[index] for index in range(grid_dimensions**n_params)],
        marker=".",
    )
    plt.colorbar()
    plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.savefig(f"../../results/scatter/{run_name}.png")
    plt.show()
    plt.clf()


def trace_on_pair(param):
    return (torch.trace(param[0]).item(), torch.trace(param[1]).item())


def proj_of_pair(i, param):
    return (param[0][i].item(), param[1][i].item())


def _plot_results(
    run_name: str,
    proj,
    min_param,
    max_param,
    grid_dimensions,
    model,
    eps=1e-3,
    title=None,
    x_label=None,
    y_label=None,
):
    """
    Plots 2D graph of speed of convergence/divergence of different parameters.
    x and y axis are defined as proj(model parameters) -> (x, y)

    Color of the point is defined by the speed, with which the model converged/diverged.

    Convergence speed = norm( #steps till loss < eps)
    Divergence speed =norm( if loss > eps: loss  )

    Parameters
    ----------
    run_name : str
        Name of the run, which data we want to plot.
    proj : function
        Function, which takes model parameters and returns 2D point
    """
    print("Warning: _print_results() is deprecated. Please use print_results() instead.")

    params_shape = model.params_shape
    n_params = n_params = model.weights.numel()

    params = param_grid(min_param, max_param, grid_dimensions, n_params)

    index_to_color_value = {}

    if title is None:
        title = f"Convergence speed of different initial parameters"

    # collect data
    print("Collecting data..")
    for index in tqdm(range(grid_dimensions ** n_params), total=grid_dimensions**n_params):
        total_loss = np.load(f"../../results/{run_name}_losses_{index}.npy")
        # check if converges
        if (total_loss < eps).any():
            # get index of first step, where loss < eps
            color = np.where(total_loss < eps)[0][0].item()
            # normalise - faster closer to 1, slower is closer to 0
            color = 1 - color / total_loss.shape[0]
            index_to_color_value[index] = color
        else:
            # get last loss
            color = -total_loss[-1]
            index_to_color_value[index] = color

        # losses.append(np.load(f"../../results/{run_name}_losses_{index}.npy"))
        # weights.append(np.load(f"../../results/{run_name}_weights_{index}.npy"))

    # get axis

    # regenerate params
    params = param_grid(min_param, max_param, grid_dimensions, n_params)

    (x, y) = zip(
        *[
            proj(param_list_to_tensor(param_list, params_shape))
            for param_list in params
        ]
    )

    # plot data
    plt.scatter(
        x,
        y,
        c=[index_to_color_value[index] for index in range(grid_dimensions**n_params)],
        marker=".",
    )
    plt.colorbar()
    plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.savefig(f"../../results/scatter/{run_name}.png")
    plt.show()
    plt.clf()


def trace_on_pair(param):
    return (torch.trace(param[0]).item(), torch.trace(param[1]).item())


def proj_of_pair(i, param):
    return (param[0][i].item(), param[1][i].item())
