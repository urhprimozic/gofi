import torch
from torch.linalg import matrix_norm
from models import GeneratorModel

def relation_loss_generator(model: GeneratorModel):
    """
    Izračuna  mean frobenius(LHS(relacija) - I) za vse relacije LHS = I.
    (npr. matrika(generator0)^2 = I)
    """
    loss = torch.tensor(0.0)
    n = len(model.group.relations)
    for relation in model.group.relations:
        loss += matrix_norm(
            model.get_matrix_of_product(relation) - torch.eye(model.dim)
        )
    return loss / n




def irr_loss_generator(model: GeneratorModel):
    """
    Izračuna (1 - ||character(model)||)**2 i.e.,
    (1 - mean_g tr**2(M_g))**2 za vse matrike M_g, ki pripadajo elementom grupe.
    """
    loss = torch.tensor(0.0)
    s = model.group.size
    for element in model.group.table:
        loss += torch.square(torch.trace(model.get_matrix_of_product(element)))
    return torch.square(loss / s - 1.0)


def unitary_loss_generator(model: GeneratorModel):
    """
    Izračuna mean frobenius(g^T g - I) za vse generatorje grupe
    """
    loss = torch.tensor(0.0)  # eye is unitary, all the products are unitary
    s = model.group.size
    n = len(model.group.generators)
    for i in range(n):
        mat = model.get_matrix_of_generator(i)
        loss += matrix_norm(
            torch.matmul(torch.transpose(mat, 0, 1), mat) - torch.eye(model.dim)
        )
    return loss / s


def loss_function_generator(model: GeneratorModel):
    return (
        irr_loss_generator(model),
        relation_loss_generator(model),  
        unitary_loss_generator(model),
    )

def triple_loss_function(model):
    '''
    Returns L_irr + L_rel + L_unitary
    '''
    irr, rel, uni = loss_function_generator(model)
    return irr + rel + uni

