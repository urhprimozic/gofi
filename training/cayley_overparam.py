from gofi.graphs.graph import random_adjacency_matrix, adjacency_matrix_cayley_Sn
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from gofi.graphs.opt import training
from gofi.models import RandomMap
import torch 
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M = torch.tensor(adjacency_matrix_cayley_Sn(5)).to(device)  # Move adjacency matrix to device


class MatrixGeneratorTransformerMLP(nn.Module):
    def __init__(self, n, dim=200, dim_feedforward=2048, depth=6, nhead=4):
        super().__init__()
        self.n = n
        self.initial = nn.Parameter(torch.rand(n, 1, dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=3,
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, n)
        )

    def forward(self):

        out = self.transformer(self.initial)
        out = out.squeeze(1)
        out = self.mlp(out)
        # out = self.mlp(out)

        return torch.reshape(out, (self.n, self.n))

class MatrixGeneratorTransformer(nn.Module):
    def __init__(self, n, dim=200, depth=3):
        super().__init__()
        self.n = n
        self.initial = nn.Parameter(torch.rand(n, 1, n))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n, nhead=4, dim_feedforward=n * 4),
            num_layers=3,
        )

    def forward(self):

        out = self.transformer(self.initial)
        out = out.squeeze(1)
        # out = self.mlp(out)

        return torch.reshape(out, (self.n, self.n))


class MatrixGenerator(nn.Module):
    def __init__(self, n, dim1=120**2, dim2=240**2):
        super().__init__()
        self.n = n
        self.initial = nn.Parameter(torch.rand(dim1))
        self.mlp = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dim2, self.n**2),
        )

    def forward(self):
        # Dummy input, since model doesn't depend on input
        out = self.mlp(self.initial)
        # return out.view(self.n, self.n)
        return torch.reshape(out, (self.n, self.n))


class MatrixGeneratorDouble(nn.Module):
    def __init__(self, n, hidden_dim=512, initial_dim=256):
        super().__init__()
        self.n = n
        self.initial = nn.Parameter(torch.rand(initial_dim))
        self.mlp = nn.Sequential(
            nn.Linear(initial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim // 2, self.n**2),
        )

    def forward(self):
        # Dummy input, since model doesn't depend on input
        out = self.mlp(self.initial)
        # return out.view(self.n, self.n)
        return torch.reshape(out, (self.n, self.n))


class CoordMLP(nn.Module):
    def __init__(self, n, hidden_dim=256):
        super().__init__()
        self.n = n
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self):
        # Build coordinate grid
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, self.n),
                torch.linspace(0, 1, self.n),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(
            -1, 2
        )  # shape: [n*n, 2]

        values = self.mlp(coords).view(self.n, self.n)
        return values


class ProjIntoRelu(nn.Module):
    def __init__(self, n, m, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = nn.Parameter(torch.rand((n, m)) * 2 - 1)
        self.B = nn.Parameter(torch.rand((m, n)) * 2 - 1)
        # ReLu, GeLu, CeLu naredijo veliko ničel in imaš obupne štartne parametre
        self.activation = nn.Softmin(dim=1)
        self.C = nn.Parameter(torch.rand((n, n)) * 2 - 1)
        self.D = nn.Parameter(torch.rand((n, n)) * 2 - 1)

    def forward(self):
        return self.activation(self.A @ self.B + self.C) @ self.D


# create transformer model
dim = 1200
inner_model = MatrixGeneratorTransformerMLP(120, dim=dim, dim_feedforward=4*200**2, depth=8, nhead=8).to(device)
#toy: inner_model = MatrixGeneratorTransformerMLP(120, dim=10, dim_feedforward=10, depth=1, nhead=1).to(device)
f = RandomMap(120, inner_model=inner_model).to(device)  # Move model to device


# hacky hacky
def transformer_schedule(warmup_steps, d_model):
    def lr_lambda(step):
        if step == 0:
            return 1e-8  # avoid division by zero
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return lr_lambda





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_steps", type=int, help="Number of steps of training")
    parser.add_argument("filename", type=str, help="Filename to save the results")
    args = parser.parse_args()

    training(
        f,
        M,
        M,
        eps=-1,
        max_steps=int(args.max_steps),
        adam_parameters={"lr": 0.001},
        grad_clipping=1000,
        verbose=100,
        B=10,
        scheduler=LambdaLR,        
        scheduler_parameters={'lr_lambda': transformer_schedule(warmup_steps=1000, d_model=dim)},
        scheduler_input=None,
    )
    # save the model 
    torch.save(f, args.filename)
