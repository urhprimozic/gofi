import torch
from torch import optim
from gofi.graphs.recursive.model import PermutationGenerator
from gofi.graphs.recursive.loss import reinforce_loss
from gofi.graphs.graph import random_adjacency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(generator : PermutationGenerator, M1, M2, eps=1e-3, max_steps=10000, batch_size=100, lr=1e-3, verbose=1):
    baseline = torch.tensor(0.0, device=device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    for step in range(max_steps):
        optimizer.zero_grad()
        loss, avg_loss = reinforce_loss(generator, M1, M2, baseline, batch_size=batch_size)
        loss.backward()
        optimizer.step()
        scheduler.step(avg_loss)

        if (step % verbose == 1) or (verbose == 1):
            print(f"[Step {step}] E[loss]={avg_loss:.6f} baseline={baseline.item():.6f}  learning rate={scheduler.get_last_lr()}")

        if avg_loss < eps:
            print(f"Converged at step {step} with E[loss]={avg_loss:.6f}")
            break

    return generator



if __name__ == '__main__':
# Dummy podatki (n=4)
    n = 4
    M1 = random_adjacency_matrix(n)
    M2 = random_adjacency_matrix(n)

    gen = PermutationGenerator(n=n, hidden_size=20*n**2, num_layers=4).to(device)
    trained_gen = training(gen, M1, M2, eps=1e-4, max_steps=2000, verbose=20)

    # Preveri en vzorec po treningu
    P, logp = trained_gen(batch_size=1)
    print("Permutacijska matrika:\n", P[0])
    print("probability:", torch.exp(logp).item())

