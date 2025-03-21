'''
Code for simulation on how different initial parameters effect convergence of a model: S3 --> R

For each opt run, three differnet outcomes are expected:
- converges to id -- final parameters are close to [[1], [1]]
- converges to sing -- final parameters are close to [[-1], [-1]]
- diverges
'''
from models import GeneratorModel
from groups import demo_S3
from training import training_loop
from loss import triple_loss_function, loss_function_generator
import torch
from grid import generate_grid, param_grid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

model = GeneratorModel(demo_S3, 1)
# 50x50 grid. Max n of iterations in opt is 400 - SHOULD BE ENOUGH BASED ON TESTS (convergence usually happens between 150 and 250 for this set of parameters)
run_name, min_param, max_param, grid_dim = generate_grid(model, triple_loss_function, -0.5,0.5, 100,400, lr=0.01, weight_decay=1e-4)
print((run_name, min_param, max_param, grid_dim))


