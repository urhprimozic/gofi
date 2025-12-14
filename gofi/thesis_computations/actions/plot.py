import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
from tqdm import tqdm
import torch
from gofi.thesis_computations.actions.compute import get_group, N

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_cyclic(Ps_initial, Ps_final,title,filename,  remove_axes=True):
    vmin = min(Ps_initial.min(), Ps_final.min())
    vmax = max(Ps_initial.max(), Ps_final.max())

    fig, axs = plt.subplots(1, 2, figsize=(16, 16))

    im0 = axs[0].imshow(Ps_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    axs[0].set_title("Začetni parametri $P_s$")

    im1 = axs[1].imshow(Ps_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    axs[1].set_title("Končni parametri $P_s$")

    # odstrani osi (po želji)
    if remove_axes:
        for ax in axs.ravel():
            ax.axis("off")

    # skupen naslov
    fig.suptitle(title, fontsize=14)

    # enoten colorbar
    cbar = fig.colorbar(im1, ax=axs, shrink=0.85)
    #cbar.set_label("Vrednost")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

def plot_dihedral(Pr_initial, Pr_final, Ps_initial, Ps_final,title,filename, remove_axes=True):
    vmin = min(Pr_initial.min(), Ps_initial.min(), Pr_final.min(), Ps_final.min())
    vmax = max(Pr_initial.max(), Ps_initial.max(), Pr_final.max(), Ps_final.max())

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    im00 = axs[0, 0].imshow(Ps_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Začetni parametri $P_s$")

    im01 = axs[0, 1].imshow(Ps_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("Končni parametri $P_s$")

    im10 = axs[1, 0].imshow(Pr_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Začetni parametri $P_r$")

    im11 = axs[1, 1].imshow(Pr_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("Končni parametri $P_r$")

    # odstrani osi (po želji)
    if remove_axes:
        for ax in axs.ravel():
            ax.axis("off")

    # skupen naslov
    fig.suptitle(title, fontsize=14)

    # enoten colorbar
    cbar = fig.colorbar(im11, ax=axs, shrink=0.85)
    #cbar.set_label("Vrednost")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()


# save everything
for group_name in ["dihedral", "cyclic"]:
    for n in tqdm(range(5, N), total=N-5):
        for m in range(5, n):
            group = get_group(group_name, n)

            if group_name == "cyclic":
                Ps_initial = torch.load(f"./results/initial_{group.name}_n={n}_m={m}.pt").to(device)
                Ps_final = torch.load(f"./results/final_{group.name}_n={n}_m={m}.pt").to(device)
                plot_cyclic(
                    Ps_initial,
                    Ps_final,
                    title=f"$C_n \\curvearrowright [{m}]$",
                    filename=f"./results/plot_{group.name}_n={n}_m={m}.png",
                )
            else:
                data_initial = torch.load(f"./results/initial_{group.name}_n={n}_m={m}.pt")
                Pr_initial = data_initial["r"].to(device)
                Ps_initial = data_initial["s"].to(device)

                data_final = torch.load(f"./results/final_{group.name}_n={n}_m={m}.pt")
                Pr_final = data_final["r"].to(device)
                Ps_final = data_final["s"].to(device)

                plot_dihedral(
                    Pr_initial,
                    Pr_final,
                    Ps_initial,
                    Ps_final,
                    title=f"$D_n \\curvearrowright [{m}]$",
                    filename=f"./results/plot_{group.name}_n={n}_m={m}.png",
                )
