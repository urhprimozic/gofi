from typing import Iterable, Hashable
import torch
import torch.nn as nn 
from gofi.models import RandomMap
from functools import reduce
import operator

def multiply_matrices(*tensors):
    return reduce(operator.matmul, tensors)

class Group:
    def __init__(self, generators : Iterable[Hashable], relations : Iterable[Iterable[Hashable]]):
        """
        Creates a new group =<generators | relations>.

        Parameters
        -----------
        generators : Iterable[Hashable]
            List (or any other iterable) of generators. Can be arbitrary types. int or str advised.
        relations : Iterable[Iterable[Hashable]]
            List (or any other iterable) of relations. Each relation is an iterator over generators.
        
        Example
        -----------
        """
        self.generators = generators
        self.relations = relations
        self.n_relations = len(list(relations))

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
        self.group = group 
        self.n = n
        # creates a dictionary between generators and their RandomModels
        self.rm = nn.ModuleDict({s : RandomMap(n) for s in group.generators})
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
        ans = torch.zeros((self.n, self.n))
        for r in self.group.relations:
            # get stochastic matrix 
            P_r = multiply_matrices(*[self.P(s) for s in r])
            log = torch.log(P_r + eps)
            ans += torch.trace(log)
        ans *= -1/self.group.n_relations
        return ans 
    