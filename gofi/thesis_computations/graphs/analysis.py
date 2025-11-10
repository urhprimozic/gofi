import matplotlib.pyplot as plt 
import pickle
import tqdm
import torch
import io
import gofi.plot.colors as gc
from comparison import run_nn
import os
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom unpickler that maps CUDA tensors to CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def add_nn_into_comparison(run_name : str):
    # collect dataset
    with open(f'./results/dataset_{run_name}.pkl', 'rb') as f:
        dataset = CPU_Unpickler(f).load() #pickle.load(f)
    # collect graphs and results
    random_graphs, cayley_graphs, all_graphs  = dataset
    for index, (graph_type, M1, Q, M2) in tqdm.tqdm(enumerate(all_graphs), total=len(all_graphs)):
        # send to device 
        M1 = M1.to(device)
        M2 = M2.to(device)
        Q = Q.to(device)
        try:
            with open(f"./results/results_{run_name}_{index}_{graph_type}.pkl" , "rb") as f:
                #results = pickle.load( f)
                results = CPU_Unpickler(f).load()
                #check if nn results are present
                if "nn" in results:
                    continue
            # run nn comparison
            results_nn = run_nn(M1, Q, M2)
            # add to results 
            results["nn"] = results_nn
            # save back
            with open(f"./results/results_{run_name}_{index}_{graph_type}.pkl" , "wb") as f:
                pickle.dump(results, f)
        except:
            print(f"Error at index: {index}. Skipping.")

def collect_results(run_name : str):
    # collect dataset
    with open(f'./results/dataset_{run_name}.pkl', 'rb') as f:
        dataset = CPU_Unpickler(f).load() #pickle.load(f)
    # collect graphs and results
    _, _, all_graphs  = dataset
    all_results = []
    for index, (graph_type, M1, Q, M2) in tqdm.tqdm(enumerate(all_graphs), total=len(all_graphs)):
        try:
            with open(f"./results/results_{run_name}_{index}_{graph_type}.pkl" , "rb") as f:
                #results = pickle.load( f)
                results = CPU_Unpickler(f).load()
                #check if nn results are present

            all_results.append(results)
        except:
            print(f"Error at index: {index}. Skipping.")
    return all_graphs, all_results

def join_results(run_names):
    all_graphs = None
    all_results = []
    for run_name in run_names:
        graphs, results = collect_results(run_name)
        if all_graphs is None:
            all_graphs = graphs
        all_results.extend(results)
    return all_graphs, all_results



def loss_on_size(all_results, filename):
    '''
    Points in 2d space of n_vertices * loss. Each method has its own color.

    For every method (vanilla, vanilla_it, nn_it) and for every graph pair (M1, M2), there is a point (M1.shape[0], loss(method, M1, M2)), colored by method's color.
    '''
    # prepare n_vertices and loss 
    n_vertices = []
    loss_vanilla_it = []
    loss_vanilla = []
    loss_nn_it = []
    loss_nn = []

    for results in all_results:
        # size
        M1, _, _ = results["graph_tuple"]
        n = M1.shape[0]
        n_vertices.append(n)
        # loss 
        loss_vanilla_it.append(results["vanilla_it"]["final_loss"])
        loss_vanilla.append(results["vanilla"]["final_loss"])
        loss_nn_it.append(results["nn_it"]["final_loss"])
        if "nn" in results:
            loss_nn.append(results["nn"]["final_loss"])
        else:
            print("Warning: nn results missing!")

    fig, ax = plt.subplots(ncols=1)
    ax.scatter(n_vertices, loss_vanilla_it, c=gc.lightorange, label="$\mathbb{R}^n$", marker='D')#, fillstyle='left')
    ax.scatter(n_vertices, loss_vanilla, c = gc.lightblue, label="$S_n$", marker='o')#, fillstyle='right')
    ax.scatter(n_vertices, loss_nn_it, c = gc.black, label="$S_n$ + nn", marker='P')#, fillstyle='full')
    ax.scatter(n_vertices, loss_nn, c = gc.darkorange, label="$$\mathbb{R}^n$ + nn", marker='X')#, fillstyle='full')

    plt.legend()
    plt.savefig(f"{filename}.pdf")

    fig, ax = plt.subplots(ncols=1)
    ax.scatter(n_vertices, loss_vanilla_it, c=gc.orange, label="$\mathbb{R}^n$", marker='D')#, fillstyle='left')
    ax.scatter(n_vertices, loss_nn_it, c = gc.black,s=5, label="$S_n$ + nn", marker='P')#, fillstyle='full')

    plt.legend()
    plt.savefig(f"{filename}_it.pdf")

def scan_and_join_results(results_dir: str = "./results", verbose: bool = False):
    """
    Scan results_dir for dataset_<run_name>.pkl files, extract run_names,
    then join their collected results.

    Returns
    -------
    run_names : list[str]
    all_graphs : list
    all_results : list
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    run_names = []
    pattern = re.compile(r"^dataset_(.+)\.pkl$")

    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if m:
            run_name = m.group(1)
            run_names.append(run_name)
            # add nn
            add_nn_into_comparison(run_name)

    run_names = sorted(set(run_names))
    if verbose:
        print(f"Found run_names: {run_names}")

    if not run_names:
        return [], None, []

    all_graphs, all_results = join_results(run_names)
    return run_names, all_graphs, all_results



