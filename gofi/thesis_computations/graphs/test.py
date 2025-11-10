from comparison import run_nn_it, run_vanilla, run_vanilla_it, run_nn
from dataset import create_dataset
from gofi.graphs.inversion_table.probs import PermModel
from gofi.models import RandomMap, ToMatrix
from gofi.graphs.inversion_table.models import PermDistConnected, LinearNN
import torch 
from gofi.graphs.loss import (
    BijectiveLoss,
    RelationLoss,
    BijectiveLossMatrix,
    RelationLossMatrix, LossGraphMatching, LossGraphMatchingRandomMap
)
from analysis import scan_and_join_results, loss_on_size


run_names, all_graphs, all_results = scan_and_join_results()

loss_on_size(all_results, "loss_on_size_all_methods")