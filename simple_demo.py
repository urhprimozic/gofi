# Looking for a function "y = a exp(-b x) + c sin(d x)"

from typing import final
import tqdm
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F


def create_test_data(a: float, b: float, c: float, d: float):
    x = torch.linspace(0, 10, 1000)
    y = a * torch.exp(-b * x) + c * torch.sin(d * x)
    y += torch.distributions.Normal(0, 0.1).sample(y.shape)
    return x, y


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization."""

    def __init__(self):
        super().__init__()
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        self.weights = nn.Parameter(weights)

    def forward(self, xs):
        a, b, c, d = self.weights
        return a * torch.exp(-b * xs) + c * torch.sin(d * xs)


def training_loop(x, y, model, optimizer, n=1000):
    losses = []
    for _ in tqdm.trange(n):
        preds = model(x)
        loss = F.mse_loss(preds, y).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


def main(n):
    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    x, y = create_test_data(2, 0.1, 3, 0.5)
    losses = training_loop(x, y, m, opt, n=n)
    plt.figure(figsize=(14, 7))
    plt.plot(losses)
    a, b, c, d = m.weights.detach().numpy()
    plt.title(f"Losses (final weights: a = {a:.2f} b = {b:.2f} c = {c:.2f} d = {d:.2f})")
    plt.show()

    with torch.no_grad():
        final_predictions = m(x)
    plt.figure(figsize=(14, 7))
    plt.plot(x, y, label="ground truth", alpha=0.5)
    plt.plot(x, final_predictions, label="predictions", alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(10**4)
