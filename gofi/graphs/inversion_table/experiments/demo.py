import gofi.graphs.inversion_table.opt as opt
from gofi.graphs.graph import (
    random_adjacency_matrix,
    random_permutation_matrix,
    permutation_matrix_to_permutation,
    permutation_to_permutation_matrix,
)
import torch
from gofi.graphs.inversion_table.models import PermDistConnected, PermDistDissconnected
from gofi.graphs.inversion_table.probs import PermModel
from gofi.graphs.inversion_table.loss import norm_loss_normalized
from torch.optim.lr_scheduler import ReduceLROnPlateau



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "__main__" == __name__:
    n = 10
    layer_size = 200
    lr = 0.001
    T = 10
    verbose = 1
    # SETTINGS for AdamPerturbed
    lr = 0.001
    noise_scale = 0.01
    grad_threshold = 0.006
    cooldown_steps = 10
    decay = 0.99

    M1 = random_adjacency_matrix(n)
    Q = random_permutation_matrix(n)
    M2 = Q.T @ M1 @ Q

    model = PermDistConnected(n, 4, layer_size, T=T).to(device)
    dist = PermModel(model, n)
    opt_params = {
        "lr": lr,
        "noise_scale": noise_scale,
        "grad_threshold": grad_threshold,
        "cooldown_steps": cooldown_steps,
        "decay": decay,
    }
    scheduler = ReduceLROnPlateau
    scheduler_parameters = {
        "mode": "min",
        "factor": 0.5,
        "patience": 300,
        "min_lr": 1e-5,
    }
    scheduler_input = "loss"
    losses = opt.training(
        dist,
        M1,
        M2,
        loss_function=norm_loss_normalized,
        max_steps=15000 if n >= 10 else 5000,
        adam_parameters=opt_params,
        scheduler=scheduler,
        scheduler_parameters=scheduler_parameters,
        scheduler_input=scheduler_input,
        grad_clipping=2,
        verbose=verbose,
        eps=0.001,
        grad_eps=None,
        adam_version="noise",
    )

    with torch.no_grad():
        print("Final loss after training:", norm_loss_normalized(dist, M1, M2).item())
        print("Most probable permutation:", dist.most_probable_permutation())
        print("Target permutation:", (torch.argmax(Q, axis=1) + 1).tolist())
        mpp = permutation_to_permutation_matrix(dist.most_probable_permutation())

        print(
            "|target(M1) - most_probable(M1)|: ",
            torch.norm(M2 - mpp.T @ M1 @ mpp).item(),
        )
