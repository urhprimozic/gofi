# Looking for a function "F: S3 ---> GLn, such that ..."

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.linalg import matrix_norm
import numpy as np
import time


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cpu > cuda za naš trashass use case
device = torch.device("cpu")
print("Using device:", device)

class GeneratorGroup:
    def __init__(
        self,
        generators: list[tuple[int]],
        table: list[list[int]],
        relations: list[list[int]]
    ) -> None:
        self.generators = generators
        self.table = table
        self.relations = relations
        self.size = len(table)

    def __str__(self) -> str:
        return (
            "Group with generators "
            + str(self.generators)
            + " and table "
            + str(self.table)
        )


class S3Group23(GeneratorGroup):
    def __init__(self) -> None:
        # id, 12, 13, 23, 123, 132
        raise NotImplementedError
        # super().__init__(
        #     [(1, 2), (1, 2, 3)],
        #     # [[], [0], [0, 1], [1, 0], [1], [1, 1]]
        # )


class S3Group22(GeneratorGroup):
    def __init__(self) -> None:
        # id, 12, 13, 23, 123, 132
        super().__init__(
            [(1, 2), (1, 3)],
            [[], [0], [1], [0, 1, 0], [0, 1], [1, 0]],
            [[0, 0], [1, 1], [0, 1, 0, 1, 0, 1]]
        )


class GeneratorModel(nn.Module):
    def __init__(
        self,
        group: GeneratorGroup,
        matrix_size: int,
        init_hint: torch.Tensor | None = None,
    ):
        """
        Parameters store the current matrices for each generator in a block matrix

        [matrix for generator 0 | matrix for generator 1 | ...]

        """
        super().__init__()
        self.group = group
        self.m_size = matrix_size
        if init_hint is None:
            weights = torch.distributions.Normal(0.0, 0.1).sample(
                (matrix_size, len(self.group.generators) * matrix_size)
            ).to(device)
        else:
            weights = init_hint
        self.weights = nn.Parameter(weights)

    def forward(self, xs):
        return xs

    def get_matrix_for_element(self, element: int):
        return self.weights[:, element * self.m_size : (element + 1) * self.m_size]

    def get_matrix_of_product(self, product: list[int]):
        if not product:
            return torch.eye(self.m_size)
        else:
            matrix = self.get_matrix_for_element(product[0])
            for i in product[1:]:
                matrix = torch.matmul(matrix, self.get_matrix_for_element(i))
            return matrix

    @staticmethod
    def get_matrix_of_element_static(product: list[int], block_matrix: np.ndarray):
        w = block_matrix.shape[0]
        matrix = np.eye(w)
        for element in product:
            matrix = matrix @ block_matrix[:, element * w : (element + 1) * w]
        return matrix


def morphism_loss_generator(model: GeneratorModel):
    raise NotImplementedError


def relation_loss_generator(model: GeneratorModel):
    """
    Izračuna  mean frobenius(LHS(relacija) - I) za vse relacije LHS = I.
    (npr. matrika(generator0)^2 = I)
    """
    loss = torch.tensor(0.0).to(device)
    n = len(model.group.relations)
    for relation in model.group.relations:
        loss += matrix_norm(
            model.get_matrix_of_product(relation) - torch.eye(model.m_size).to(device)
        )
    return loss / n


def irr_loss_generator(model: GeneratorModel):
    """
    Izračuna (1 - ||character(model)||)**2 i.e.,
    (1 - mean_g tr**2(M_g))**2 za vse matrike M_g, ki pripadajo elementom grupe.
    """
    loss = torch.tensor(0.0).to(device)
    s = model.group.size
    for element in model.group.table:
        loss += torch.square(torch.trace(model.get_matrix_of_product(element)))
    return torch.square(loss / s - 1.0)


def unitary_loss_generator(model: GeneratorModel):
    """
    Izračuna mean frobenius(g^T g - I) za vse generatorje grupe
    """
    loss = torch.tensor(0.0).to(device)  # eye is unitary, all the products are unitary
    s = model.group.size
    n = len(model.group.generators)
    for i in range(n):
        mat = model.get_matrix_for_element(i)
        loss += matrix_norm(
            torch.matmul(torch.transpose(mat, 0, 1), mat) - torch.eye(model.m_size).to(device)
        )
    return loss / s


def loss_function_generator(model: GeneratorModel):
    return (
        irr_loss_generator(model),
        relation_loss_generator(model),  # morphism_loss_generator(model),
        unitary_loss_generator(model),
    )


def training_loop(model, optimizer, n, a, b, c, eps=1e-3, verbose=False):
    losses = []
    for i in range(n):
        i_loss, m_loss, u_loss = loss_function_generator(model)
        components = [i_loss.item(), m_loss.item(), u_loss.item()]
        loss = a * i_loss + b * m_loss + c * u_loss
        components.append(loss.item())
        losses.append(components)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # check for convergence
        if loss  < eps:
            if verbose:
                print(f"Converged at step {i}")
                return np.array(losses).T
    if verbose:
        print(f"Failed to converge after {n} steps. Last loss = {loss.item()}")
    return np.array(losses).T


def main(n, a, b, c, matrix_size=2, lr=0.001):
    g = S3Group22()
    m = GeneratorModel(
        g, matrix_size
    )
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    losses = training_loop(m, opt, n, a, b, c)
    plt.figure(figsize=(14, 7))
    for loss, name in zip(losses, ["irr", "rel", "unitary", "w_total"]):
        plt.plot(loss, label=name)
    plt.legend()
    plt.title(f"Losses (a = {a}, b = {b}, c = {c})")
    appendix = f"{int(time.time())}"
    plt.savefig(f"results/s3_losses{appendix}.png")
    final_weights = m.weights.detach().numpy()
    with open(f"results/final{appendix}.txt", "w") as f:
        print(f"abc = {a, b, c}", file=f)
        print(f"lr = {lr}", file=f)
        print(
            f"final losses: {losses[:, -1].tolist()} (irr, rel, uni, w_tot)", file=f
        )
        print(g, file=f)
        for i, element in enumerate(g.table):
            e_weights = GeneratorModel.get_matrix_of_element_static(
                element, final_weights
            ).tolist()
            print(f"{i}: {e_weights}", file=f)


if __name__ == "__main__":
    main(10**4, 1, 2, 2)
