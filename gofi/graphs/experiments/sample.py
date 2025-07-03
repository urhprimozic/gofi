import torch
import numpy as np
from gofi.models import RandomMap
from gofi.graphs.loss import BijectiveLoss, RelationLoss
from gofi.graphs.graph import random_adjacency_matrix, adjacency_matrix_cayley_Sn
from math import factorial
from gofi.graphs.opt import training
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange, blue, cmap_dakblue_blue
import networkx as nx



torch.set_printoptions(precision=2, sci_mode=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_graph_image(adj,  filename="graph.png",title=None, layout="spring", dpi=300):
    """
    Save a graph from an adjacency matrix as an image.

    Parameters
    ----------
    adj : np.ndarray or torch.Tensor
        Adjacency matrix of the graph (must be square).
    filename : str
        Output filename (should end in .png or .pdf).
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'random'.
    dpi : int
        DPI for saved image.
    """
    # Convert to NumPy if it's a torch tensor
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()

    # Ensure square matrix
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    
    n_vertices = adj.shape[0]

    # Create graph
    G = nx.from_numpy_array(adj)

    # Choose layout
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "random": nx.random_layout,
    }
    pos = layout_funcs.get(layout, nx.spring_layout)(G)

    # Draw and save
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color= cmap_dakblue_blue(np.linspace(0, 1, n_vertices)), 
        edge_color="gray",
        node_size=600,
        font_size=10,
    )
    plt.axis("off")
    #plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()




def sample_cayley(n_iterator, sample_size, run_name=None):
    if run_name is None:
        run_name=""
    else:
        run_name = run_name + "_"

    for n in n_iterator:
        graph_name=run_name + f"cayley_graph_{n}"
        M = adjacency_matrix_cayley_Sn(n).to(device)
        save_graph_image(M, filename=graph_name + ".pdf")


        for sample in range(sample_size):

            f = RandomMap(factorial(n)).to(device)
            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)
            loss = RelationLoss(f, M, M) + BijectiveLoss(f)
            loss = round(float(loss.clone().cpu().detach().numpy()), 3)
            plt.title(f"Loss: {loss}")
            plt.axis("off")
            #plt.colorbar()
            plt.savefig(f"{graph_name}_sample_{sample}_initial.pdf", bbox_inches='tight')
            plt.close()

            training(
                f, M, M, adam_parameters={"lr": 0.001}, max_steps=500000, eps=1e-12
            )

            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)
            #plt.tight_layout()

            loss = RelationLoss(f, M, M) + BijectiveLoss(f)
            loss = round(float(loss.clone().cpu().detach().numpy()), 3)
            plt.title(f"Loss: {loss}")

            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"{graph_name}_sample_{sample}_final.pdf", bbox_inches='tight')
            plt.close()


def sample_random(n_iterator, sample_size, run_name=None):
    if run_name is None:
        run_name=""
    else:
        run_name = run_name + "_"

    for index, n in enumerate(n_iterator):

        M = random_adjacency_matrix(n).to(device)
        
        graph_name=run_name + f"random_graph_{index+1}_on_{n}"

        save_graph_image(M, filename=graph_name)

        for sample in range(sample_size):

            f = RandomMap(n).to(device)
            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)
           # plt.tight_layout()


            # get loss 
            loss = RelationLoss(f, M, M) + BijectiveLoss(f)
            loss = round(float(loss.clone().cpu().detach().numpy()), 3)
            
            plt.title(f"Loss: {loss}")

            plt.axis("off")
            #plt.colorbar()
            plt.savefig(graph_name + f"sample_{sample}_initial.pdf", bbox_inches='tight')
            plt.close()


            # train
            training(
                f, M, M, adam_parameters={"lr": 0.001}, max_steps=500000, eps=1e-12
            )

            plt.imshow(f.P().clone().cpu().detach(), cmap=blueorange)
            #plt.tight_layout()

            loss = RelationLoss(f, M, M) + BijectiveLoss(f)
            loss = round(float(loss.clone().cpu().detach().numpy()), 3)

            plt.title(f"Loss: {loss}")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(graph_name + f"sample_{sample}_final.pdf", bbox_inches='tight')
            plt.close()


def sample_random_plotless(n_iterator, sample_size):
    filenames = {}

    for index, n in enumerate(n_iterator):

        name = f"graph_{index+1}_on_{n}.pt"

        M = random_adjacency_matrix(n).to(device)
        # save graph
        M.save(name)

        filenames_samples = []

        for sample in range(sample_size):

            f = RandomMap(n).to(device)
            sample_name = f"graph_{index + 1}_on_{n}_sample_{sample}"
            filenames_samples.append(sample_name)
            # save initial values
            f.save(f"{sample_name}_initial.pt")

            training(
                f, M, M, adam_parameters={"lr": 0.001}, max_steps=500000, eps=1e-12
            )
            # save final values
            f.save(f"{sample_name}_final.pt")
        filenames[name] = filenames_samples
    return filenames

def plot_sampled_graphs(filenames, title=None, main_title=None, layout="spring"):
    """
    Plot sampled graphs from filenames.
    
    Parameters
    ----------
    filenames : dict
        Dictionary with filenames of sampled graphs.
    title : str
        Title for the plot.
    main_title : str
        Main title for the plot.
    """
    
    for name, samples in filenames.items():
        # collect graph 
        adj = torch.load(name)


        for sample_name in samples:
            fig, axs = plt.subplots(1, len(samples), figsize=(15, 5))
            # plot: GRAPH | INITIAL P | FINAL P

            # plot graph------------------------------------
             # Convert to NumPy if it's a torch tensor
            if isinstance(adj, torch.Tensor):
                adj = adj.detach().cpu().numpy()

            # Create graph
            G = nx.from_numpy_array(adj)

            # Choose layout
            layout_funcs = {
                "spring": nx.spring_layout,
                "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout,
                "random": nx.random_layout,
            }
            pos = layout_funcs.get(layout, nx.spring_layout)(G)

            # Draw and save
            plt.figure(figsize=(6, 6))
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="blue",
                edge_color="gray",
                node_size=850,
                font_size=20,
                ax=axs[0]
            )
            axs[0].set_axis_off()
            axs[0].set_title("Graph")
            # plot initial P--------------------------------
            P_initial = torch.load(f"{sample_name}_initial.pt").P().cpu().detach
            axs[1].imshow(P_initial, cmap=blueorange)
            axs[1].set_title("Initial P")
            axs[1].axis("off")
            # plot final P----------------------------------
            P_final = torch.load(f"{sample_name}_final.pt").P().cpu().detach
            axs[2].imshow(P_final, cmap=blueorange)
            axs[2].set_title("Final P")
            axs[2].axis("off")
    raise NotImplementedError("Plotting sampled graphs is not implemented yet.")
