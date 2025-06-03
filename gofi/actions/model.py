from typing import Iterable, Hashable
import torch
import torch.nn as nn 
from gofi.models import RandomMap
from functools import reduce
import operator
from gofi.graphs.loss import BijectiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def multiply_matrices(*tensors):
    return reduce(operator.matmul, tensors)

class Group:
    def __init__(self, generators : Iterable[Hashable], relations : Iterable[Iterable[Hashable]], name=None):
        """
        Creates a new group =<generators | relations>.

        Parameters
        -----------
        generators : Iterable[Hashable]
            List (or any other iterable) of generators. Can be arbitrary types. int or str advised.
        relations : Iterable[Iterable[Hashable]]
            List (or any other iterable) of relations. Each relation is an iterator over generators.
        name : str | None 
            group name
        Example
        -----------
        """
        self.name  = name
        self.generators = generators
        self.relations = relations
        self.n_relations = len(list(relations))
        self.n_generators = len(list(generators))

class ActionModel(nn.Module):
    
    def __init__(self, group : Group, n : int):
        """
        Creates a new ActionModel, which models a mapping from a group into the distributions over fun([n], [n]).

        Parameters
        ----------
        group : Group
            Group, which is being modeled.
        n : int
            Number of elements the group is actiong on.
        
        """
        super().__init__()
        self.group = group 
        self.n = n
        # creates a dictionary between generators and their RandomModels
        self.rm = nn.ModuleDict({s : RandomMap(n).to(device) for s in group.generators})
    def P(self, s : Hashable):
        """
        Returns stochastic matrix of a random map for a generator s

        Parameters
        s : Hashable
            generator
        """
        f =  self.rm[s]
        return f.P()

    def relation_loss(self, eps=1e-10):
        """
        Return relation loss of the model, defined as

        1/|R| sum tr(log(P_r + eps)) for r in relations

        where P_r is the stohastic matrix of a random map, defined with r.
        """
        ans = 0
        for r in self.group.relations:
            # get stochastic matrix 
            P_r = multiply_matrices(*[self.P(s) for s in r])
            log = torch.log(P_r + eps)
            ans += torch.trace(log)
        ans *= -1/self.group.n_relations
        return ans 
    
    def bijective_loss(self, eps=1e-10):
        ans = 0
        for s in self.group.generators:
            f  =self.rm[s]
            ans += BijectiveLoss(f, eps=eps)
        ans /= self.group.n_generators
        return ans
    