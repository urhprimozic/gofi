from torch.models import GeneratorModel, GeneratorGroup
from math import factorial

demo_S3 = GeneratorGroup(
            generators=[(1, 2), (1, 3)],
            relations=[[0, 0], [1, 1], [0, 1, 0, 1, 0, 1]],
             table = [[], [0], [1], [0, 1, 0], [0, 1], [1, 0]]
        )


demo_C3 = GeneratorGroup(
            generators=["s"],
            relations=[[0,0,0]],
             table = [[], [0], [0,0]]
        )

def coxeter_presentation(n : int):
    '''
    Returns a coxeter presetation of group Sn

    Returns
    ----------
    Sn : GeneratorGroup
        coxeter presentation
    '''
    raise NotImplementedError("TODO : create table (disjoin cycles; cycle to generator)")
    generators = [(i, i+1) for i in range(1, n)]
    size = factorial(n)
    
    # every trasposition squared is 1 
    involution_relations = [[i, i] for i in range(len(generators))]

    #commutativity for nonadjecent swaps
    comm_relations = [[i, j, i, j] for i in range(len(generators)) for j in range(len(generators)) if (abs(i-j) >= 2) and (i < j) ]

    # s_i s_(i+1) s_i =  s_(i+1) s_i  s_(i+1)
    braid_relations = [[i, i+1, i, i+1, i, i+1] for i in range(len(generators) - 1)] 
    
    # all relations
    relations = involution_relations + comm_relations + braid_relations

    # table of all elements

    

def get_S_n(n):
    return coxeter_presentation(n)
