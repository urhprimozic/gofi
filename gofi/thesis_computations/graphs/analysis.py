import numpy as np
import matplotlib.pyplot as plt 
import pickle
import tqdm
import torch
import io
import gofi.plot.colors as gc
from comparison import run_nn
import os
import re
import hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom unpickler that maps CUDA tensors to CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def add_nn_into_comparison(run_name : str, override=True, print_error=False):
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
                if "nn" in results and not override:
                    continue
            # run nn comparison
            results_nn = run_nn(M1, Q, M2)
            # add to results 
            results["nn"] = results_nn
            # save back
            with open(f"./results/results_{run_name}_{index}_{graph_type}.pkl" , "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error at index: {index}. Skipping.")
            if print_error:
                print(e)
            continue

def collect_results_deprecated(run_name : str):
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







def scan_and_join_results(results_dir: str = "./results", verbose: bool = False, nn=False):
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
            if nn:
                add_nn_into_comparison(run_name)

    run_names = sorted(set(run_names))
    if verbose:
        print(f"Found run_names: {run_names}")

    if not run_names:
        return [], None, []

    all_graphs, all_results = join_results(run_names)
    return run_names, all_graphs, all_results

def plot_loss_over_time(method_to_results, output_filename, suptitle="", loss_key="relation_losses", log_scale=True):
    '''
    Expects a dictionary {label : results, 'vanilla_it' : results, ...}
    Plot loss over time for different methods.'''
    plt.figure()
    for label, results in method_to_results.items():
        #print(label)
        losses = results[loss_key]
        plt.plot(losses, label=label)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Korak")
    plt.ylabel("Napaka")
    plt.suptitle(suptitle, fontsize=8)
    plt.title(f"Napaka različnih metod skozi čas")
    plt.legend()
    plt.savefig(f"./results/loss_over_time_{output_filename}.pdf")
    plt.close()
    


def plot_hyperparams_nn_vs_vanilla(skip_nn_vanilla=False):
    #with open("./results/hyperparams_graphs.pkl", "rb") as f:
     #   graph_tuples = pickle.load(f)
    final_losses_nn=[]
    final_losses_vanilla=[]

    nn_losses= []
    #for graph_index, (M1, Q, M2) in enumerate(graph_tuples):
    index=0
    for n in hyperparams.vertices:
     #""   n = M1.shape[0]
        for params in hyperparams.parameters_list:
            index += 1
            try:
                with open(
                    f"./results/hyperparams_n{n}_ns{params['noise_scale']}_gt{params['grad_threshold']}_cd{params['cooldown_steps']}_decay{params['decay']}.pkl",
                    "rb",
                ) as f:
                    result = pickle.load(f)
            except Exception as e:
                print(f"Error at n: {n} with params: {params}. Skipping.")
                print(e)
                break
            if not skip_nn_vanilla:
                plot_loss_over_time({"Nevronske mreže": result["nn"],"Brez": result["vanilla"]} , f"test_{index}_n{n}", suptitle=f"Graf z  {n} vozljišči\nNastavitve: noise_scale={params['noise_scale']}, grad_threshold={params['grad_threshold']}, cooldown_steps={params['cooldown_steps']}, decay={params['decay']}")
            # store final losses 
            nn_losses.append( result["nn"]["losses"])
            results_nn = result["nn"]
            results_vanilla = result["vanilla"]
            final_losses_nn.append( results_nn["final_loss"])
            final_losses_vanilla.append( results_vanilla["final_loss"])
           
    # average
    print("Average final loss nn:", sum(final_losses_nn)/len(final_losses_nn))
    print("Average final loss vanilla:", sum(final_losses_vanilla)/len(final_losses_vanilla))

    # plot nn losses over time
    
def plot_hyperparams_nn_vs_nn(log_scale=True, mute_errors=True):
    nn_losses= []
    best = []
    cmap = plt.get_cmap('gist_rainbow')
    cmap2 = plt.get_cmap('inferno')
    colors = [cmap(i) for i in np.linspace(0.2, 0.9, 20)] + [cmap2(i) for i in np.linspace(0, 1, 16)]
    index=0
    for n in hyperparams.vertices:
        params_to_final_loss = {}
        final_loss_to_params = {}
        plt.figure()
        for params in hyperparams.parameters_list:
            index += 1
            try:
                with open(
                    f"./results/hyperparams_n{n}_ns{params['noise_scale']}_gt{params['grad_threshold']}_cd{params['cooldown_steps']}_decay{params['decay']}.pkl",
                    "rb",
                ) as f:
                    result = pickle.load(f)
                    # add loss to plot 
                    graph_of_loss = result["nn"]["losses"]
                    params_to_final_loss[(params['noise_scale'], params['grad_threshold'], params['cooldown_steps'], params['decay'])] = result["nn"]["final_loss"]
                    final_loss_to_params[result["nn"]["final_loss"]] = (params['noise_scale'], params['grad_threshold'], params['cooldown_steps'], params['decay'])
                    plt.plot(graph_of_loss,color=colors[index-1], label=f"ns={params['noise_scale']}, gt={params['grad_threshold']}, cd={params['cooldown_steps']}, decay={params['decay']}")

            except Exception as e:
                if mute_errors:
                    continue
                print(f"Error at n: {n} with params: {params}. Skipping.")
                continue
        # save plot
        if log_scale:
            plt.yscale("log")
        plt.xlabel("Korak")
        plt.ylabel("Napaka")
        plt.suptitle(f"Graf z  {n} vozljišči", fontsize=7)
        plt.title(f"Napaka nevronskih mrež skozi čas za različne nastavitve hiperparametrov")
        plt.legend(fontsize=6  )
        plt.savefig(f"./results/hyperparams_nn_{index}_n{n}.pdf")
        plt.close()

        best.append((params_to_final_loss, final_loss_to_params))
    for pf, fp in best:
        try:
            min_loss = min(fp.keys())
            print(f"Best final loss: {min_loss} with params: {fp[min_loss]}")
        except:
            continue


def collect_results(run_name=None):
    '''
    Scans ./results for files. Expects files, saved by running compparison.py
    Returns list of dictionaries with results.

    if run_name is given, only files starting with results_{run_name} are considered.
    Multiple run names can be given in side a iterable
    '''
    if run_name is None:
        run_name=[""]
    elif isinstance(run_name, str):
        run_name = [run_name] 

    path = "./results"
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # remove bad names 
    tuple_run_names = ("results_{name}" for name in run_name)

    filenames = [f for f in filenames if f.startswith(tuple_run_names)]

    list_of_results = []

    for filename in tqdm.tqdm(filenames, total=len(filenames)):
        
        full_path = os.path.join(path, filename)
        try:
            with open(full_path, "rb") as f:
                results = CPU_Unpickler(f).load() #TODO a to kej zjeba??!
                list_of_results.append(results)
        except Exception as e:
            print(f"Error reading {filename}, skipping.")
            print(e)
            continue
    return list_of_results


def loss_on_size(list_of_results,filename, methods = ["vanilla_it", "vanilla", "nn_it", "nn"], markers = ['.', '.', '+', 'x']):
    '''
    Points in 2d space of n_vertices * loss. Each method has its own color.

    For every method (vanilla, vanilla_it, nn_it) and for every graph pair (M1, M2), there is a point (M1.shape[0], loss(method, M1, M2)), colored by method's color.
    '''
    all_methods = ["vanilla_it", "vanilla", "nn_it", "nn", "mild_nn_it"]

    # prepare n_vertices and loss 
    n_vertices = []
    loss = {
        "vanilla_it" : [],
        "vanilla" : [],
        "nn_it" : [],
        "nn" : [],

    }
    for results in list_of_results:
        # get n
        M1, _, _ = results["graph_tuple"]
        n = M1.shape[0]
        n_vertices.append(n)
        # final RELATION    loss
        for method in methods:
            if method not in all_methods:
                raise ValueError(f"Unknown method: {method}")
            loss[method].append(results[method]["relation_losses"][-1])

    fig, ax = plt.subplots(ncols=1)

    labels = {
        "vanilla_it" : "Tabela inverzij",
        "vanilla" : "Brez tabele inverzij",
        "nn_it" : "Nevronske mreže in tabela inverzij",
        "nn" : "Nevronske mreže brez tabele inverzij",
    }

    colors = [gc.lightblue, gc.darkorange, gc.black, gc.lightorange]
  
    for method, color, marker in zip(all_methods, colors, markers):
        if method in methods:
            ax.scatter(n_vertices, loss[method], c=color, label=labels[method], marker=marker, alpha=0.8)
        
      
    plt.legend()
    plt.xlabel("Število vozlišč")
    plt.ylabel("Napaka")
    plt.title("Napake različnih metod glede na velikosti grafov")
    plt.savefig(f"{filename}.pdf")
    plt.close()


def average_loss_on_size(list_of_results,filename, methods = ["vanilla_it", "vanilla", "nn_it", "nn", "mild_nn_it"], markers = ['.', '.', '+', 'x', '.']):
    '''
    Points in 2d space of n_vertices * loss. Each method has its own color.

    For every method (vanilla, vanilla_it, nn_it) and for every graph size n, there is a point (n, E[loss(method)]), colored by method's color.
    '''
    all_methods = ["vanilla_it", "vanilla", "nn_it", "nn", "mild_nn_it"]

    # prepare n_vertices and loss 
    
    
    loss = {
        "vanilla_it" : {},
        "vanilla" : {},
        "nn_it" : {},
        "nn" : {},
        "mild_nn_it" : {},

    }
    for results in list_of_results:
        # get n
        M1, _, _ = results["graph_tuple"]
        n = M1.shape[0]

        # final RELATION    loss
        for method in methods:
            if method not in all_methods:
                raise ValueError(f"Unknown method: {method}")
            # adds new loss to n
            if n not in loss[method]:
                loss[method][n] = []
            loss[method][n] = [results[method]["relation_losses"][-1]] + loss[method][n]
            
    # average
    for method in methods:
        for n in loss[method]:
           # print(method, " - ", n, " - ", loss[method][n])
            loss[method][n] = sum(loss[method][n]) / len(loss[method][n])

    fig, ax = plt.subplots(ncols=1)

    labels = {
        "vanilla_it" : "Tabela inverzij",
        "vanilla" : "Brez tabele inverzij",
        "nn_it" : "Nevronske mreže in tabela inverzij",
        "nn" : "Nevronske mreže brez tabele inverzij",
        "mild_nn_it" : "Blaga overparametrizacija in tabela inverzij",
    }

    colors = [gc.lightblue, gc.darkorange, gc.black, gc.lightorange, gc.green]
  
    for method, color, marker in zip(all_methods, colors, markers):
        if method in methods:
            # collect n and loss
            n_vertices = list(loss[method].keys())
            loss_values = [loss[method][n] for n in n_vertices]
            ax.scatter(n_vertices, loss_values, c=color, label=labels[method], marker=marker, alpha=0.8)
        
      
    plt.legend()
    plt.xlabel("Število vozlišč")
    plt.ylabel("Napaka")
    plt.title("Napake različnih metod glede na velikosti grafov")
    plt.savefig(f"average_{filename}.pdf")
    plt.close()

def main():
    # collect results
    list_of_results = collect_results()
    # plot loss on size
    average_loss_on_size(list_of_results, "loss_on_size_vanilla_vs_it", methods=["vanilla_it", "vanilla"],markers=["o","o","o","o"])
    average_loss_on_size(list_of_results, "loss_on_size_vanilla_vs_it_vs_nn_it", methods=["vanilla_it", "vanilla", "nn_it"],markers=["o","o","o","o"])

def lor_plot_losses(lor, output_filename,methods, suptitle="", loss_key="relation_losses", log_scale=True):
    '''
    Plots list of results losses. Combine with collect results
    '''
    for index, all_results in enumerate(lor):
        labels_to_results = {
            method : all_results[method] for method in methods
        }
        of = output_filename + "_" + str(index)

        # suptitle 
        M1, _, _ = all_results["graph_tuple"]
        suptitle = f"Graf z {M1.shape[0]} vozlišči"
        
        plot_loss_over_time(labels_to_results, output_filename=of, suptitle=suptitle, loss_key=loss_key, log_scale=log_scale)
