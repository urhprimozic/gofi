from gofi.ode.dihedral.gradient_flow import GradientFlow
import numpy as np
from tqdm import tqdm
import argparse
from gofi.ode.dihedral.loss import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_n", type=str, help="Number of different dihedral groups tested")
    parser.add_argument("n_samples", type=int, help="Number of samples, generated for each group")
    parser.add_argument("filename", type=str, help="Filename to save the results")
    args = parser.parse_args()

    dihedral_to_solutions={}
    for n in tqdm(range(1, int(args.max_n) + 1), desc="Iterating over dihedral groups", total=int(args.max_n)):
        # create new equations
        irr_loss = IrreducibilityLoss(n)
        unit_loss = UnitaryLoss()
        rel_loss = RelationLoss(n)
        loss = irr_loss + unit_loss + rel_loss
        
        gf = GradientFlow(2, loss, clipping_limit=10)

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
        

