import matplotlib.pyplot as plt 
import pickle
import tqdm

def collect_results(run_name : str):
    # collect dataset
    with open(f'dataset_{run_name}.pkl', 'rb') as f:
        dataset = pickle.load(f)
    # collect graphs and results
    _, _, all_graphs  = dataset
    all_results = []
    for index, (graph_type, M1, Q, M2) in tqdm.tqdm(enumerate(all_graphs), total=len(all_graphs)):
        with open(f"./results/results_{run_name}_{index}_{graph_type}.pkl" , "rb") as f:
            results = pickle.load( f)
        all_results.append(results)
    return all_graphs, all_results

