# TODO
# functions for creating grid of different parameters and running gf.solve() for each one
# KAR COPILOTU REČ, DA NAJ SPIŠE

import numpy as np
from gofi.ode.dihedral.gradient_flow import GradientFlow
from tqdm import tqdm


def grid(min_value : float , max_value : float , resolution : int, dim : int, gradient_flow : GradientFlow, verbose : bool = False, eps=0.0001, t_max=50):
    """
    Creates a grid of all possible pairs of dim*dim matrices (R, S) and runs GradientFlow.solve(R, S) for each pair.

    Parameters:
    -----------
    min_value : float
        Minimum value for the grid.
    max_value : float
        Maximum value for the grid.
    resolution : int
        Number of points in the linspace for each matrix element.
    dim : int
        Dimension of the matrices R and S.
    gradient_flow : GradientFlow
        An instance of the GradientFlow class.
    verbose : bool
        If True, shows a progress bar.
    eps : float
        Tolerance for the solver. Solving stops, if norm of the gradient is smaller than eps.
    t_max : float
        Maximum time for the solver. Solving stops, if t > t_max.

    Returns:
    --------
    (flattened, solutions) : tuple
        A tuple containing the flattened grid of matrices and the solutions for each pair of matrices.
    """
    # Create a linspace for the grid
    values = np.linspace(min_value, max_value, resolution)
    # n_parameters = dim * dim * 2
    # meshgrid
    grid = np.meshgrid(*[values] * (dim * dim * 2))
    # unpack grid to array
    stacked = np.stack(grid, axis=0)
    # stack different matrix elements next to each other
    points = np.moveaxis(stacked, 0, -1)
    # Reshape into pairs of matrices
    matrices = points.reshape(*points.shape[:-1], 2, 2, 2)

    # get array of pairs [R, S] 
    flattened = matrices.reshape(-1, 2, 2, 2)

    # solve gradient flow for each pair of initial values R,S in flattened

    iterator = flattened
    if verbose:
        iterator = tqdm(flattened, desc="Solving Gradient Flow", total=flattened.shape[0])

    solutions = []
    
    # Solve gradient flow for each point in grid
    for R0, S0 in iterator:
        # solve d(R,S) = -gradL(R,S)dt for initial values R(0)=R0, S(0)=S0
        solutions.append( gradient_flow.solve(R0, S0) )
        

    return flattened, solutions

def save_grid(filename, grid_results):
    """
    Saves the grid of matrices and the solutions to a file.

    Parameters:
    -----------
    filename : str
        Name of the file to save the data to.
    grid_results : tuple
        A tuple containing the flattened grid of matrices and the solutions for each pair of matrices.
    """
    # Save the data to a file
    np.save(filename, grid_results, allow_pickle=True)