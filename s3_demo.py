"""
Demo za S3. Irr(S3) = {1, sign, ro},
kjer je ro tak kot v https://urbanjezernik.github.io/teorija-upodobitev/#zgled-1.11.

trenutno je tole v demo stanju. Namesto klasične regresije funkcija izgube pove,
koliko stran smo od nerazcepne uporobitve.
(norma karakterja upodobitve more biti 1 in zadeva mora biti homomorfizem).

Koda za učenje sledi matejevi:
    - x je vektor vrednosti v input layerju, ki predstavljajo elemente grupe,
    - y so dejanski elementi grupe. Velja, da je y[i] tisti element,
      ki je v našem vektorskem zapisu* enak x[i]

*elemente grupe moramo zakodirati v vektorje. Očitne kode so recimo g_i -> e_i. 
S3 ima 6 elementov --> input layer mora biti element R^6 
"""
from sympy.combinatorics.named_groups import (
    SymmetricGroup,
)  # oni množijo cikle iz leve!
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.linalg import matrix_norm
import tqdm
import matplotlib.pyplot as plt



class VectorCode:
    """
    Elemente grupe zakodira v (ortogonalne?) vektorje, ki služijo kot input za našo mrežo.

    Atributi
    -------
    - `.group` - vrne `sympy` grupo
    - `.group_elements` - vrne elemente, urejene v seznam.
    - `vectors` - vrne kode elementov iz group_elements
    - `element` - slovar, ki pretvori element grupe v njegovo kodo
    - `vector` - inverz za `element`
    """

    def __init__(self, group) -> None:
        self.group = group
        self.group_elements = list(group.elements)
        self.vectors = [
            torch.tensor([0.0] * i + [1.0] + [0.0] * (group.order() - (i + 1)))
            for i in range(group.order())
        ]  # enotski vektorji
        self.element = {v: e for e, v in zip(self.group_elements, self.vectors)}
        self.vector = {e: v for e, v in zip(self.group_elements, self.vectors)}

        # potrebujemo tabelo  produktov [g*h], ki jo preslikamo z modelom
        products_elements = [
            g * h for g in self.group_elements for h in self.group_elements
        ]
        products_vectors = [self.vector[g] for g in products_elements]


# demo mreža
class DemoModel(torch.nn.Module):
    def __init__(self, input_size, matrix_size):
        super().__init__()
        torch.manual_seed(420)
        hidden_layer_size = 5
        output_size = matrix_size**2
        self.matrix_size = matrix_size
        self.lin1 = nn.Linear(input_size, hidden_layer_size)
        self.lin2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.lin3 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = torch.reshape(x, (self.matrix_size, self.matrix_size))
        return x


def MorphismLoss(model, vector_code: VectorCode):
    """
    SUM(  (f(gh) - f(g)f(h))**2)

    Parameters
    --------
    - `model` -
    """
    L = torch.tensor(0.0)
    for g, gv in zip(vector_code.group_elements, vector_code.vectors):
        for h, hv in zip(vector_code.group_elements, vector_code.vectors):
            gh = vector_code.vector[g * h]
            Fgh = model(gh)
            FgFh = torch.matmul(model(gv), model(hv))
            L += matrix_norm(Fgh - FgFh)
    return L


def IrrLoss(model: DemoModel, vector_code: VectorCode):
    """
    Izračuna (||character(model)|| - 1)**2
    """
    L = torch.tensor(0.0)
    for g in vector_code.vectors:
        L += torch.trace(model(g)) ** 2
    order = torch.tensor(float(vector_code.group.order()))
    return (L / order - torch.tensor(1.0)) ** 2


def unitaryLoss(model: DemoModel, vector_code: VectorCode):
    """
    Izračuna SUM ||model(g)model(g)^* - 1 ||
    """
    L = torch.tensor(0.0)
    for g in vector_code.vectors:
        mg = model(g)
        mg_star = torch.t(mg)  # transpose
        i = torch.matmul(mg, mg_star)
        L += matrix_norm(i - torch.eye(2))
    return L


def lossFunction(model, vector_code: VectorCode, a=1, b=10, c=1):
    return (
        a * IrrLoss(model, vector_code)
        + b * MorphismLoss(model, vector_code)
        + c * unitaryLoss(model, vector_code)
    )


def training_loop(model, optimizer, vector_code, n=1000):
    losses = []
    for _ in tqdm.trange(n):
        # loss = F.mse_loss(preds, y).sqrt()
        loss = lossFunction(model, vector_code)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


def character_norm(model):
    """
    Izračuna (||character(model)||
    """
    L = torch.tensor(0.0)
    for g in vector_code.vectors:
        L += torch.trace(model(g)) ** 2
    order = torch.tensor(float(vector_code.group.order()))
    return L / order


if __name__ == "__main__":
    n = 10**4
    print("TODO - for zanke spremeni v operacije s  tenzorji")
    print("WARNING : trenutno dela le nad R !")
    vector_code = VectorCode(SymmetricGroup(3))
    m = DemoModel(6, 2)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)

    losses = training_loop(m, opt, vector_code, n=n)
    plt.figure(figsize=(14, 7))
    plt.plot(losses)
    plt.title("Losses")
    plt.show()

    with torch.no_grad():
        print("Final function: ")
        for g, gv in zip(vector_code.group_elements, vector_code.vectors):
            print(f"{g} maps to  \n{m(gv)}\n")
