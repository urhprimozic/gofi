import pickle
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
from tqdm import tqdm
import torch
from gofi.thesis_computations.actions.compute import get_group, N

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_cyclic(
    Ps_initial, Ps_final, loss, permutation, title, filename, remove_axes=True
):
    vmin = min(Ps_initial.min(), Ps_final.min())
    vmax = max(Ps_initial.max(), Ps_final.max())

    # 2 plot columns + 1 colorbar column
    fig = plt.figure(constrained_layout=False, figsize=(8, 4))
    gs = fig.add_gridspec(
        nrows=2, ncols=3, width_ratios=[1, 1, 0.05], height_ratios=[0.01, 1]
    )
    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    im0 = ax0.imshow(Ps_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax0.set_title(
        "Začetni parametri $P_s$",
    )
    im1 = ax1.imshow(Ps_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax1.set_title(f"$\\overlineP_s = {permutation}$")

    if remove_axes:
        ax0.axis("off")
        ax1.axis("off")

    fig.suptitle(title + f"; končna napaka: {loss}", fontsize=16)
    fig.colorbar(im1, cax=cax)
    # Tighten layout: reduce bottom/top padding and inter-axes spacing
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.96, wspace=0.05)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_dihedral(
    Pr_initial,
    Pr_final,
    Ps_initial,
    Ps_final,
    loss,
    permutation_r,
    permutation_s,
    title,
    filename,
    remove_axes=True,
):
    vmin = min(Pr_initial.min(), Ps_initial.min(), Pr_final.min(), Ps_final.min())
    vmax = max(Pr_initial.max(), Ps_initial.max(), Pr_final.max(), Ps_final.max())

    # 2x2 plot grid + 1 colorbar column
    fig = plt.figure(constrained_layout=False, figsize=(8, 6))
    gs = fig.add_gridspec(
        nrows=2, ncols=3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1]
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])  # colorbar spans both rows

    im00 = ax00.imshow(Ps_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax00.set_title("Začetni parametri $P_s$", pad=2)
    im01 = ax01.imshow(Ps_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax01.set_title(f"\\overlineP_s = {permutation_s}", pad=2)

    im10 = ax10.imshow(Pr_initial, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax10.set_title("Začetni parametri $P_r$", pad=2)
    im11 = ax11.imshow(Pr_final, cmap=blueorange, vmin=vmin, vmax=vmax)
    ax11.set_title(f"$\\overlineP_r = {permutation_r}$", pad=2)

    if remove_axes:
        for ax in (ax00, ax01, ax10, ax11):
            ax.axis("off")

    fig.suptitle(title + f"; končna napaka: {loss}", fontsize=13)
    fig.colorbar(im11, cax=cax)

    # Reduce column spacing and bottom whitespace
    fig.subplots_adjust(
        top=0.90, bottom=0.02, left=0.02, right=0.96, wspace=0.06, hspace=0.06
    )

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def read_and_plot(group_name, n, m, test=False):
    group = get_group(group_name, n)
    if group_name == "cyclic":
        data_initial = (
            torch.load(f"./results/initial_{group.name}_n={n}_m={m}.pt").cpu().detach()
        )
        Ps_initial = data_initial["z"].cpu().detach()

        data_final = (
            torch.load(f"./results/final_{group.name}_n={n}_m={m}.pt").cpu().detach()
        )
        Ps_final = data_final["z"].cpu().detach()
        permutation = data_final["mode"]

        with open(f"./results/loss_{group.name}_n={n}_m={m}.pkl", "rb") as f:
            loss = pickle.load(f)
            loss = round(loss, 2)
        filename = f"./results/plot_{group.name}_n={n}_m={m}.png"
        if test:
            filename = f"./results/TEST_plot_{group.name}_n={n}_m={m}.png"
        plot_cyclic(
            Ps_initial,
            Ps_final,
            loss,
            permutation,
            title=f"$C_{{{n}}} \\curvearrowright [{m}]$",
            filename=filename,
        )
    else:
        data_initial = torch.load(f"./results/initial_{group.name}_n={n}_m={m}.pt")
        Pr_initial = data_initial["r"].cpu().detach()
        Ps_initial = data_initial["s"].cpu().detach()

        data_final = torch.load(f"./results/final_{group.name}_n={n}_m={m}.pt")
        Pr_final = data_final["r"].cpu().detach()
        Ps_final = data_final["s"].cpu().detach()
        permutation_r = data_final["r_mode"]
        permutation_s = data_final["s_mode"]
        with open(f"./results/loss_{group.name}_n={n}_m={m}.pkl", "rb") as f:
            loss = pickle.load(f)
            loss = round(loss, 2)
        filename = f"./results/plot_{group.name}_n={n}_m={m}.png"
        if test:
            filename = f"./results/TEST_plot_{group.name}_n={n}_m={m}.png"
        plot_dihedral(
            Pr_initial,
            Pr_final,
            Ps_initial,
            Ps_final,
            loss,
            permutation_r,
            permutation_s,
            title=f"$D_{{{n}}} \\curvearrowright [{m}]$",
            filename=filename,
        )


test = False

if __name__ == "__main__":
    if test:
        read_and_plot("dihedral", 6, 5, test=True)
        read_and_plot("cyclic", 6, 5, test=True)
    else:
        # save everything
        for group_name in ["dihedral", "cyclic"]:
            for n in tqdm(range(5, N), total=N - 5):
                for m in range(max(5, n - 5), n + 1):

                    read_and_plot(group_name, n, m)
