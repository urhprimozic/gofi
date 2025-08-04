import torch
import matplotlib.pyplot as plt
from s3_descent_generators import S3Group22, GeneratorModel, training_loop, device
from tqdm import tqdm
from datetime import datetime
import numpy as np
import itertools

def param_monte_carlo(n_points, n_params, run_name=None):
        """
    Create a randomized grid of parameters to optimize over. 
    run_name is used as seed.

    Returns
    -------
    Iterator over tuples, each tuple is a different set of initial parameters.
    """
    

def param_grid(min_param, max_param, n_points, n_params):
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


def param_list_to_tensor(initial_params_list, n_generators, matrix_size):
    initial_params = torch.tensor(initial_params_list).to(device)
    # matej petković:
    initial_params = torch.reshape(
        initial_params, (matrix_size, n_generators * matrix_size)
    )
    # good way:
    #  initial_params = torch.reshape(
    #          initial_params, (n_generators, matrix_size, matrix_size)
    #      )
    return initial_params

def run_monte_carlo(
    min_param,
    max_param,
    n_generated_points,
    opt_steps=2000,
    a=1,
    b=1,
    c=1,
    matrix_size=2,
    lr=0.001,
    opt="adam",
    run_name=None,
    eps=1e-3,
    verbose=False
):
    """
    Runs a SGD/adam for each set of parameters, which are choosen at random.
    Number of all generated points is n_generated_points.

    Parameters
    ----------
    min_param : int


    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting run {run_name}..")

    # TODO move this to parameters of function
    g = S3Group22()
    n_generators = 2
    n_params = n_generators * matrix_size**2

    # generate grid of all different parameters
    params = param_grid(min_param, max_param, n_points, n_params)

    # n_points**n_params                    x           2           x   matrix_size   x   matrix_size
    # all different settings of parameters  x  2 different matrices x   matrix_size   x   matrix_size

    # run optimization for each set of parameters
    for index in tqdm(range(n_generated_points), total=n_generated_points):
        # get initial params
        initial_params = torch.rand((matrix_size, n_generators * matrix_size)).to(device)
        # rescape between min and max 
        initial_params = min_param + (max_param - min_param) * initial_params
        # save initial params
        torch.save(initial_params, f"results/{run_name}_mc_{index}.pt")

        # create model with initial parameters
        m = GeneratorModel(g, matrix_size, init_hint=initial_params)
        # optimizer
        if opt == "adam":
            opt = torch.optim.Adam(m.parameters(), lr=lr)
        else:
            opt = torch.optim.SGD(m.parameters(), lr=lr)

        # run optimization
        losses = training_loop(m, opt, opt_steps, a, b, c, eps=eps, verbose=verbose)

        # save data
        # save final losses
        np.save(f"results/{run_name}_losses_{index}.npy", losses)
        # save final weights
        final_weights = m.weights.cpu().detach().numpy()
        np.save(f"results/{run_name}_weights_{index}.npy", final_weights)
    return run_name


def plot_mc_results(
    run_name: str,
    proj,
    min_param,
    max_param,
    matrix_size,
    n_generators,
    n_generated_points,
    n_params,
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
    
    index_to_color_value = {}

    if title is None:
        title = f"Convergence speed of different initial parameters"

    # collect data
    print("Collecting data..")

    x = []
    y = []

    for index in tqdm(range(n_generated_points), total=n_generated_points):
        # get initial params
        initial_params = torch.load(f"results/{run_name}_mc_{index}.pt")
        
        dx, dy = proj(initial_params)
        x.append(dx)
        y.append(dy)
        

        # get losses
        total_loss = np.load(f"results/{run_name}_losses_{index}.npy")[:, -1]
        # check if converges
        if (total_loss < eps).any():
            # get index of first step, where loss < eps
            color = np.where(total_loss < eps)[0][0].item()
            # normalise - faster closer to 1, slower is closer to 0
            color = (1 - color / total_loss.shape[0])*2
            index_to_color_value[index] = color
        else:
            # get last loss
            color = -total_loss[-1]
            index_to_color_value[index] = color

        # losses.append(np.load(f"results/{run_name}_losses_{index}.npy"))
        # weights.append(np.load(f"results/{run_name}_weights_{index}.npy"))

    # get axis
    
    # read params
   
    # plot data
    plt.scatter(
        x, y, c=[index_to_color_value[index] for index in range(n_generated_points)],
        marker="."
    )
    plt.colorbar()
    plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(f"results/scatter/{run_name}.png")  
    plt.show()

def opt_run(
    min_param,
    max_param,
    n_points,
    opt_steps=2000,
    a=1,
    b=1,
    c=1,
    matrix_size=2,
    lr=0.001,
    opt="adam",
    run_name=None,
    eps=1e-3,
    verbose=False
):
    """
    Runs a SGD/adam for each set of parameters.
    Each parameter is bounded by min and max. There are n_points steps between min and max for each parameter.
    The number of all steps taken is n_points^n_parameters = n_points^8

    Parameters
    ----------
    min_param : int


    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting run {run_name}..")

    # TODO move this to parameters of function
    g = S3Group22()
    n_generators = 2
    n_params = n_generators * matrix_size**2

    # generate grid of all different parameters
    params = param_grid(min_param, max_param, n_points, n_params)

    # n_points**n_params                    x           2           x   matrix_size   x   matrix_size
    # all different settings of parameters  x  2 different matrices x   matrix_size   x   matrix_size

    # run optimization for each set of parameters
    for index, initial_params_list in tqdm(enumerate(params), total=n_points**n_params):
        # get initial params
        initial_params = param_list_to_tensor(
            initial_params_list, n_generators, matrix_size
        )

        # create model with initial parameters
        m = GeneratorModel(g, matrix_size, init_hint=initial_params)
        # optimizer
        if opt == "adam":
            opt = torch.optim.Adam(m.parameters(), lr=lr)
        else:
            opt = torch.optim.SGD(m.parameters(), lr=lr)

        # run optimization
        losses = training_loop(m, opt, opt_steps, a, b, c, eps=eps, verbose=verbose)

        # save data
        # save final losses
        np.save(f"results/{run_name}_losses_{index}.npy", losses)
        # save final weights
        final_weights = m.weights.cpu().detach().numpy()
        np.save(f"results/{run_name}_weights_{index}.npy", final_weights)
    return run_name

def plot_results(
    run_name: str,
    proj,
    min_param,
    max_param,
    matrix_size,
    n_generators,
    n_points,
    n_params,
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
    params = param_grid(min_param, max_param, n_points, n_params)
    losses = []
    weights = []
    index_to_color_value = {}

    if title is None:
        title = f"Convergence speed of different initial parameters"

    # collect data
    print("Collecting data..")
    for index, param_list in tqdm(enumerate(params), total= n_points**n_params):
        total_loss = np.load(f"results/{run_name}_losses_{index}.npy")[:, -1]
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

        # losses.append(np.load(f"results/{run_name}_losses_{index}.npy"))
        # weights.append(np.load(f"results/{run_name}_weights_{index}.npy"))

    # get axis
    
    # regenerate params
    params = param_grid(min_param, max_param, n_points, n_params)
    
    (x, y) = zip(*[
            proj(param_list_to_tensor(param_list, n_generators, matrix_size))
            for param_list in params
        ])

    # plot data
    plt.scatter(
        x, y, c=[index_to_color_value[index] for index in range(n_points**n_params)], marker="."
    )
    plt.colorbar()
    plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.savefig(f"results/scatter/{run_name}.png")  
    plt.show()


def trace_on_pair(param):
    # depetkovićing --> transform to normal shapes
    param = torch.reshape(param, (2, 2, 2)) 

    assert param.shape[0] == 2
    return (torch.trace(param[0]).item(), torch.trace(param[1]).item())


def proj_of_pair(i, param):
    # depetkovićing --> transform to normal shapes
    param = torch.reshape(param, (2, 2, 2)) 
    assert param.shape[0] == 2
    return (param[0][i].item(), param[1][i].item())


if __name__ == "__main__":
    min_param = -0.5
    max_param = 0.5
    n_points = 3
    opt_steps = 1500
    eps = 0.01
    lr = 0.001
    verbose=False


    n_generated_points = 1500


    run = "monte_carlo"
    #run = "grid"

    if run == 'monte_carlo':
        run_name= "2025-03-17 20:32:43"
        if False:
            run_name = run_monte_carlo(
                min_param=min_param,
                max_param=max_param,
                n_generated_points=n_generated_points,
                opt_steps=opt_steps,
                a=1,
                b=1,
                c=2,
                matrix_size=2,
                lr=lr,
                #run_name="test1",
                eps=eps,
                verbose=verbose
            )
        plot_mc_results(
            run_name=run_name,
            #run_name="test1",
            proj=trace_on_pair,
            min_param=min_param,
            max_param=max_param,
            matrix_size=2,
            n_generators=2,
            n_generated_points=n_generated_points,
            n_params=8,
            title="Speed of convergence/divergence",
            x_label="trace of (1 2)",
            y_label="trace of (1 3)",
            eps=0.01    )

    if run == 'grid':
        run_name = opt_run(
            min_param=min_param,
            max_param=max_param,
            n_points=n_points,
            opt_steps=opt_steps,
            a=1,
            b=1,
            c=2,
            matrix_size=2,
            lr=lr,
            #run_name="test1",
            eps=eps,
            verbose=True
        )
        plot_results(
            run_name=run_name,
            #run_name="test1",
            proj=trace_on_pair,
            min_param=min_param,
            max_param=max_param,
            matrix_size=2,
            n_generators=2,
            n_points=n_points,
            n_params=8,
            title="Speed of convergence/divergence",
            x_label="trace of (1 2)",
            y_label="trace of (1 3)",
            eps=0.01    )
