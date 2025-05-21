from gofi.ode.dihedral.gradient_flow import GradientFlow
import numpy as np
from tqdm import tqdm

def sample_fixed_char(min_value, max_value, character):
    """
    Somehow (don't ask how) creates random matrix with character=character. min_calue and max_value are parameters, that do something.

    Works semi ok (=fast enough)
    """
    X = np.random.rand(2,2) * (max_value - min_value) + min_value
    
    #alternativa - prbl enako hitro
    #s = np.random.rand()*2-1
   # X[0][0] = s * character 
   # X[1][1] = (1 - s) * character

    X[0][0] = character  - X[1][1]
    return X

def grid(
    min_char : int,
    max_char : int,
    resolution: int,
    n_samples : int,
    min_value: float,
    max_value: float,
    gradient_flow: GradientFlow,
    verbose: bool = False,
    eps=0.0001,
    t_max=50,
):
    """
    Creates resolution * resolution grid of different characters for R and S. 
    For each pair of characters (charR, charS), n_samples random matrices of such characters are sampled.
    Each pair of sampled matrices (R0,S0) is used as a initial value for Gradient Flow. 

    Returns:
    --------
    (characters, samples) : tuple
        A tuple containing the characters of the matrices and the solutions for each pair of matrices.

        characters : list
            A list of tuples containing the characters (charR, charS) for each pair of matrices.
        samples : list
            A list of dictionaries containing the initial points and solutions for each pair of matrices.
            Each dictionary contains:
                - 'init_points' : list of tuples (R0, S0) for each sample
                - 'solutions' : list of solutions for each sample
            Samples[i] corresponds to characters[i].
    """
    # Create a linspace for the grid
    values = np.linspace(min_char, max_char, resolution)
    # meshgrid
    charsR, charsS = np.meshgrid(*[values] * 2)
    
    # solve gradient flow for each pair of initial values R,S in flattened

    iterator = zip(charsR.flatten(), charsS.flatten())

    if verbose:
        iterator = tqdm(
            iterator, desc="Solving Gradient Flow", total=resolution**2
        )

    characters = []
    samples = []

    # Solve gradient flow for each point in grid
    for charR , charS in iterator:
        characters.append((charR, charS))
        # sample different initial values
        solutions = []
        init_points = []
        for i in range(n_samples):
            # get new sample
            R0 = sample_fixed_char(min_value, max_value, charR)
            S0 = sample_fixed_char(min_value, max_value, charS)
            # get the trajectory of gradient flow 
            solution = gradient_flow.solve(R0, S0, eps, t_max)
            # save 
            init_points.append((R0, S0))
            solutions.append(solution)
        sample = {'init_points' : init_points, 'solutions' : solutions}
        samples.append(sample)

    return characters, samples
