# run this file 
# to get the plots of the actions in the thesis
from gofi.actions.model import Group, ActionModel
import matplotlib.pyplot as plt
from gofi.plot.colors import blueorange
from gofi.actions.opt import training
from tqdm import tqdm
import torch
import pickle

#from gofi.actions.experiment.sample import plot_model_cyclic, plot_model_dihedral

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 20

def get_group(group_name, n):
    if group_name == "cyclic":
        return Group("z", ["z" * n], name=f"C{n}")
    else:
        return Group(generators="rs", relations=["r" * n, "rsrs", "ss"], name=f"D{n}")

def loss_function(model: ActionModel):
    return  model.relation_loss() +  model.bijective_loss()

def save_model(model, group, filename):
    if group.name[0] == "C":
        torch.save(model.P("z").clone().cpu().detach(), filename)
    else:
        torch.save({
            "r": model.P("r").clone().cpu().detach(),
            "s": model.P("s").clone().cpu().detach()
        }, filename)


fix = False
if __name__ == "__main__":

    for group_name in ["dihedral", "cyclic"]:
        for n in tqdm(range(5, N), total=N-5):
            for m in range(5, n+1):
                if fix:
                    if m < n:
                        continue
                group = get_group(group_name, n)



                model = ActionModel(group,  m).to(device)
                
                save_model(model, group, f"./results/initial_{group.name}_n={n}_m={m}.pt")


                training(model, eps=0.001, max_steps=2000, adam_parameters={"lr":0.01}, verbose=0)

                loss = loss_function(model).item()

                with open(f"./results/loss_{group.name}_n={n}_m={m}.pkl", "wb") as f:
                    pickle.dump(loss, f)

                save_model(model, group, f"./results/final_{group.name}_n={n}_m={m}.pt")

