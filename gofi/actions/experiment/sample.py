# big simulation for different groups
from gofi.actions.model import Group, ActionModel
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
from gofi.actions.opt import training
from tqdm import tqdm

def plot_model_cyclic(model, title : str, main_title : str,  filename):
    

    plt.imshow(model.P("z").clone().detach(), cmap=blueorange)#"hot")
    plt.title(title +" z")
    
    plt.colorbar()

    plt.suptitle(main_title) 
    plt.savefig(filename)


def plot_model_dihedral(model, title : str, main_title : str,  filename):
    fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns

    imr = axs[0].imshow(model.P("r").clone().detach(), cmap=blueorange)#"hot")
    axs[0].set_title(title +" r")


    ims = axs[1].imshow(model.P("s").clone().detach(),cmap=blueorange)#"hot")
    axs[1].set_title(title +" s")

    plt.tight_layout()
    fig.colorbar(ims, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)
    fig.suptitle(main_title) 
    plt.savefig(filename)




cyclic = [Group("z", ["z" * n], name=f"C{n}") for n in range(2, 10)]
dihedral = [
    Group(generators="rs", relations=["r" * n, "rsrs", "ss"], name=f"D{n}") for n in range(3, 10)
]
groups = cyclic + dihedral


sample_size = 50

for group in tqdm(groups, total=len(groups)):
    if group.name[0] == "C":
        plot_model = plot_model_cyclic
    else:
        plot_model = plot_model_dihedral


    for n in range(5, 15):
        for sample in range(sample_size):
            model = ActionModel(group, n)
            plot_model(model, f"Initial parameters of generator", f"${group.name} \\to $ fun$([{{n}}])$", f"{group.name}_on_{n}_sample{sample}_initial.pdf")
            # train
            training(model, eps=0.001, max_steps=50000, adam_parameters={"lr":0.001})
            plot_model(model, f"Final parameters of generator", f"${group.name} \\to $ fun$([{{n}}])$", f"{group.name}_on_{n}_sample{sample}_final.pdf")