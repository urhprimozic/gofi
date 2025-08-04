'''
Code for simulation on how different initial parameters effect convergence of a model: S3 --> R

For each opt run, three differnet outcomes are expected:
- converges to id -- final parameters are close to [[1], [1]]
- converges to sing -- final parameters are close to [[-1], [-1]]
- diverges
'''
from gofi.autograd.models import GeneratorModel
from gofi.groups import demo_S3
from gofi.autograd.training import training_loop
from gofi.autograd.loss import triple_loss_function, loss_function_generator
import torch
from gofi.autograd.grid import generate_grid, param_grid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

model = GeneratorModel(demo_S3, 1)


if __name__ == '__main__':  
    # collect settings
    min_param = int(sys.argv[1])
    max_param = int(sys.argv[2])
    grid_dim = int(sys.argv[3])

   
    run_name, min_param, max_param, grid_dim = generate_grid(model, triple_loss_function, min_param,max_param, grid_dim,400, lr=0.01, weight_decay=1e-4)
    print((run_name, min_param, max_param, grid_dim))


