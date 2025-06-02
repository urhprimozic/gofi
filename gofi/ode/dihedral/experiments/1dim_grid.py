from gofi.ode.dihedral.gradient_flow import GradientFlow
import numpy as np
from tqdm import tqdm
import argparse
from gofi.ode.dihedral.loss import *


def grid(
    min_char : int,
    max_char : int,
    resolution: int,
    gradient_flow: GradientFlow,
    verbose: bool = False,
    eps=0.0001,
    t_max=50,
):
    """
    Creates resolution * resolution grid of different values for real numbers R and S.
    Each pair (R,S) is useda as a initial value for GradientFLow.

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_n", type=str, help="Number of different dihedral groups tested")
    parser.add_argument("n_samples", type=int, help="Number of samples, generated for each group")
    parser.add_argument("filename", type=str, help="Filename to save the results")
    parser.add_argument("--dim", default="2",type=str, help="Dimension of the modeled representation.")
    args = parser.parse_args()

    dim = int(args.dim)
    
    dihedral_to_solutions={}
    for n in tqdm(range(1, int(args.max_n) + 1), desc="Iterating over dihedral groups", total=int(args.max_n)):
        # create new equations
        irr_loss = IrreducibilityLoss(n)
        unit_loss = UnitaryLoss()
        rel_loss = RelationLoss(n)
        loss = irr_loss + unit_loss + rel_loss
        
        gf = GradientFlow(dim, loss, clipping_limit=10)

        # run simulations
        solutions = gf.solve_on_uniform_sample(
            n_samples=int(args.n_samples),
            min_value=-1,
            max_value=1,
            multiprocess=False,
            eps=0.001,
            t_max=5,
            verbose=True
        )
        # save solutions
        np.save(args.filename + f"n={n}", solutions, allow_pickle=True)
        dihedral_to_solutions[n] = solutions
    np.save(args.filename, dihedral_to_solutions, allow_pickle=True)
        