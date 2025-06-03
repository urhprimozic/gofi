# big simulation for different groups
from gofi.actions.model import Group, ActionModel
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
from gofi.actions.opt import training
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_model_cyclic(model, title : str, main_title : str,  filename):
    

    plt.imshow(model.P("z").clone().cpu().detach(), cmap=blueorange)#"hot")
    plt.title(title +" z")
    
    plt.colorbar()

    plt.suptitle(main_title) 
    plt.savefig(filename)
    plt.close()


def plot_model_dihedral(model, title : str, main_title : str,  filename):
    fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns

    imr = axs[0].imshow(model.P("r").clone().cpu().detach(), cmap=blueorange)#"hot")
    axs[0].set_title(title +" r")


    ims = axs[1].imshow(model.P("s").clone().cpu().detach(),cmap=blueorange)#"hot")
    axs[1].set_title(title +" s")

    plt.tight_layout()
    fig.colorbar(ims, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)
    fig.suptitle(main_title) 
    plt.savefig(filename)
    plt.close()



N = 50
cyclic = [Group("z", ["z" * n], name=f"C{n}") for n in range(6, 10)]
dihedral = [
    Group(generators="rs", relations=["r" * n, "rsrs", "ss"], name=f"D{n}") for n in range(3, N)
]
groups = dihedral

def loss_function(model: ActionModel):
    return  model.relation_loss() +  model.bijective_loss()

sample_size = 10

for n in tqdm(range(5, N), total=N-5):
    # get dihedral group
    group = Group(generators="rs", relations=["r" * n, "rsrs", "ss"], name=f"D{n}")

    if group.name[0] == "C":
        plot_model = plot_model_cyclic
    else:
        plot_model = plot_model_dihedral



    for sample in range(sample_size):
        model = ActionModel(group, n).to(device)
        plot_model(model, f"Initial parameters of generator", f"${group.name} \\to $ fun$([{{n}}])$", f"{group.name}_on_{n}_sample{sample}_initial.pdf")
        # train
        training(model, eps=0.001, max_steps=50000, adam_parameters={"lr":0.001}, verbose=1000)
        loss = loss_function(model)
        plot_model(model, f"Final parameters of generator", f"${group.name} \\to $ fun$([{{n}}])$\nLoss: {loss}", f"{group.name}_on_{n}_sample{sample}_final.pdf")