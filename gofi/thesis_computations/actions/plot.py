import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
from tqdm import tqdm
import torch
from gofi.thesis_computations.actions.compute import get_group, N

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_cyclic(Ps_initial, Ps_final, title, filename, remove_axes=True):
    vmin = min(Ps_initial.min(), Ps_final.min())
    vmax = max(Ps_initial.max(), Ps_final.max())

    # 2 plot columns + 1 colorbar column
    fig = plt.figure(constrained_layout=False, figsize=(8, 3.2))
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.05])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im0 = ax0.imshow(Ps_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax0.set_title("Začetni parametri $P_s$", pad=2)
    im1 = ax1.imshow(Ps_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax1.set_title("Končni parametri $P_s$", pad=2)

    if remove_axes:
        ax0.axis("off")
        ax1.axis("off")

    fig.suptitle(title, fontsize=14, y=0.98)  # minimal gap above
    fig.colorbar(im1, cax=cax)
    # Tighten layout: reduce bottom/top padding and inter-axes spacing
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.96, wspace=0.05)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_dihedral(Pr_initial, Pr_final, Ps_initial, Ps_final, title, filename, remove_axes=True):
    vmin = min(Pr_initial.min(), Ps_initial.min(), Pr_final.min(), Ps_final.min())
    vmax = max(Pr_initial.max(), Ps_initial.max(), Pr_final.max(), Ps_final.max())

    # 2x2 plot grid + 1 colorbar column
    fig = plt.figure(constrained_layout=False, figsize=(8, 6))
    gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])  # colorbar spans both rows

    im00 = ax00.imshow(Ps_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax00.set_title("Začetni parametri $P_s$", pad=2)
    im01 = ax01.imshow(Ps_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax01.set_title("Končni parametri $P_s$", pad=2)

    im10 = ax10.imshow(Pr_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax10.set_title("Začetni parametri $P_r$", pad=2)
    im11 = ax11.imshow(Pr_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax11.set_title("Končni parametri $P_r$", pad=2)

    if remove_axes:
        for ax in (ax00, ax01, ax10, ax11):
            ax.axis("off")

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.colorbar(im11, cax=cax)

    # Reduce column spacing and bottom whitespace
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.96, wspace=0.06, hspace=0.06)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # save everything
    for group_name in ["dihedral", "cyclic"]:
        for n in tqdm(range(5, N), total=N-5):
            for m in range(5, n):
                group = get_group(group_name, n)

                if group_name == "cyclic":
                    Ps_initial = torch.load(f"./results/initial_{group.name}_n={n}_m={m}.pt").cpu().detach()
                    Ps_final = torch.load(f"./results/final_{group.name}_n={n}_m={m}.pt").cpu().detach()

                    plot_cyclic(
                        Ps_initial,
                        Ps_final,
                        title=f"$C_n \\curvearrowright [{m}]$",
                        filename=f"./results/plot_{group.name}_n={n}_m={m}.png",
                    )
                else:
                    data_initial = torch.load(f"./results/initial_{group.name}_n={n}_m={m}.pt")
                    Pr_initial = data_initial["r"].cpu().detach()
                    Ps_initial = data_initial["s"].cpu().detach()

                    data_final = torch.load(f"./results/final_{group.name}_n={n}_m={m}.pt")
                    Pr_final = data_final["r"].cpu().detach()
                    Ps_final = data_final["s"].cpu().detach()

                    plot_dihedral(
                        Pr_initial,
                        Pr_final,
                        Ps_initial,
                        Ps_final,
                        title=f"$D_n \\curvearrowright [{m}]$",
                        filename=f"./results/plot_{group.name}_n={n}_m={m}.png",
                    )
