import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PermutationGenerator(nn.Module):
    def __init__(self, n, hidden_size=128, num_layers=2):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=n, hidden_size=hidden_size,
                          batch_first=True, num_layers=num_layers).to(device)
        self.output_layer = nn.Linear(hidden_size, n).to(device)

    def forward(self, batch_size=1):
        inp = torch.ones(batch_size, 1, self.n, device=device)
        h = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, device=device)

        available = torch.ones(batch_size, self.n, dtype=torch.bool, device=device)
        perm_matrix = torch.zeros(batch_size, self.n, self.n, device=device)
        log_probs = torch.zeros(batch_size, device=device)

        for step in range(self.n):
            out, h = self.gru(inp, h)
            logits = self.output_layer(out[:, -1, :])
            logits[~available] = -1e9
            probs = F.softmax(logits, dim=-1)

            idx = torch.multinomial(probs, 1).squeeze(-1)
            log_probs += torch.log(probs[torch.arange(batch_size), idx] + 1e-9)

            perm_matrix[torch.arange(batch_size), step, idx] = 1.0
            available[torch.arange(batch_size), idx] = False

            inp = F.one_hot(idx, num_classes=self.n).float().unsqueeze(1)

        return perm_matrix, log_probs


def matrix_to_permutation(P: torch.Tensor):
    """
    Pretvori permutacijsko matriko P (n x n) v permutacijo podano kot vektor n števil.
    
    Args:
        P (torch.Tensor): matrika oblike (n, n), ki je permutacijska matrika.
    
    Returns:
        perm (torch.Tensor): vektor permutacije oblike (n,), kjer perm[i] = j pomeni,
                             da je v vrstici i enica v stolpcu j.
    """
    if P.dim() != 2 or P.size(0) != P.size(1):
        raise ValueError("Matrika mora biti kvadratna permutacijska matrika.")
    
    # Najdemo indeks stolpca, kjer je v vrstici enica
    perm = torch.argmax(P, dim=1)
    return perm

