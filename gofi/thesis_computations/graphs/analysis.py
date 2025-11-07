import matplotlib.pyplot as plt 
import pickle
import tqdm
import torch
import io
import gofi.plot.colors as gc

# Custom unpickler that maps CUDA tensors to CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

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
            all_results.append(results)
        except:
            print(f"Error at index: {index}. Skipping.")
    return all_graphs, all_results

def loss_on_size(all_results):
    '''
    Points in 2d space of n_vertices * loss. Each method has its own color.

    For every method (vanilla, vanilla_it, nn_it) and for every graph pair (M1, M2), there is a point (M1.shape[0], loss(method, M1, M2)), colored by method's color.
    '''
    # prepare n_vertices and loss 
    n_vertices = []
    loss_vanilla_it = []
    loss_vanilla = []
    loss_nn_it = []

    for results in all_results:
        # size
        M1, _, _ = results["graph_tuple"]
        n = M1.shape[0]
        n_vertices.append(n)
        # loss 
        loss_vanilla_it.append(results["vanilla_it"]["final_loss"])
        loss_vanilla.append(results["vanilla"]["final_loss"])
        loss_nn_it.append(results["nn_it"]["final_loss"])

    fig, ax = plt.subplots(ncols=1)
    ax.scatter(n_vertices, loss_vanilla_it, c=gc.orange, label="$\mathbb{R}^n$", marker='D')#, fillstyle='left')
    ax.scatter(n_vertices, loss_vanilla, c = gc.lightblue, label="$S_n$", marker='o')#, fillstyle='right')
    ax.scatter(n_vertices, loss_nn_it, c = gc.black, label="$S_n$ + nn", marker='P')#, fillstyle='full')

    plt.legend()
    plt.savefig("test.pdf")

    fig, ax = plt.subplots(ncols=1)
    ax.scatter(n_vertices, loss_vanilla_it, c=gc.orange, label="$\mathbb{R}^n$", marker='D')#, fillstyle='left')
    ax.scatter(n_vertices, loss_nn_it, c = gc.black,s=5, label="$S_n$ + nn", marker='P')#, fillstyle='full')

    plt.legend()
    plt.savefig("test_it.pdf")



