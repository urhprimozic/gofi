# big simulation for different groups
from gofi.actions.model import Group, ActionModel

cyclic = [Group("z", ["z" * n]) for n in range(1, 10)]
dihedral = [
    Group(generators="rs", relations=["r" * n, "rsrs", "ss"]) for n in range(1, 10)
]
groups = cyclic + dihedral

