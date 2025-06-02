import pickle 
from gofi.ode.dihedral.gradient_flow import GradientFlow
import numpy as np
from tqdm import tqdm
import argparse
from gofi.ode.dihedral.loss import *


def grid(
    min_value : int,
    max_value : int,
    resolution: int,
    gradient_flow: GradientFlow,
    verbose: bool = False,
    eps=0.0001,
    t_max=1,
):
    """
    Creates resolution * resolution grid of different values for real numbers R and S.
    Each pair (R,S) is useda as a initial value for GradientFLow.
    """
    # Create a linspace for the grid
    values = np.linspace(min_value, max_value, resolution)
    
    # meshgrid
    points_r, points_s = np.meshgrid(*[values] * 2)
    # charsR, charsS = np.meshgrid(*[values] * 2)
    
    # solve gradient flow for each pair of initial values R,S in flattened

    iterator = zip(points_r.flatten(), points_s.flatten())

    if verbose:
        iterator = tqdm(
            iterator, desc="Solving Gradient Flow", total=resolution**2
        )

    characters = []
    samples = []

    grid_dict={}

    # Solve gradient flow for each point in grid
    for r , s in iterator:
        r0 = r.reshape((1,1))
        s0 = s.reshape((1,1))
        
        solution = gradient_flow.solve(r0, s0, eps=eps, t_max=t_max)
        grid_dict[(r.item(), s.item())] = solution

    return grid_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=str, help="Order of rotation . Encodes n at D2n.")
    parser.add_argument("min_value", type=str, help="Minimal value of the grid")
    parser.add_argument("max_value", type=str, help="Max value of the grid")
    parser.add_argument("resolution", type=str, help="Grid resolution")
    parser.add_argument("filename", type=str, help="Filename to save the result")
    parser.add_argument("--eps",default="0.01", type=str, help="Eps")
    parser.add_argument("--t_max",default="1", type=str, help="Eps")
    args = parser.parse_args()
    # collect args
    n = int(args.n)
    eps = float(args.eps)
    t_max = float(args.t_max)
    min_value = int(args.min_value)
    max_value = int(args.max_value)
    resolution = int(args.resolution)
    filename = args.filename

    # create new equations
    irr_loss = IrreducibilityLoss(n)
    unit_loss = UnitaryLoss()
    rel_loss = RelationLoss(n)
    loss = irr_loss + unit_loss + rel_loss
    gf = GradientFlow(1, loss, clipping_limit=100)
    # run grid
    grid_dict = grid(min_value, max_value, resolution, gf, verbose=True, eps=eps, t_max=t_max)

    # save grid 
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(grid_dict, f)