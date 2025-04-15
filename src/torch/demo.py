from training import training_loop
from models import GeneratorModel, GeneratorGroup
from loss import loss_function_generator, unitary_loss_generator, irr_loss_generator, relation_loss_generator
import torch
from groups import demo_S3
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import ReduceLROnPlateau

# RUN ONE OPTIMISATION
lr = 0.001

S3 = demo_S3
model = GeneratorModel(S3, 2)
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(opt, 'min')

def loss_function(model):
    irr, rel, uni = loss_function_generator(model)
    return irr + rel + uni

def extract_info(model):
    irr, rel, uni = loss_function_generator(model)
    return irr.item(), rel.item(), uni.item(), (irr + rel + uni).item()

# train network
losses, infos, converged = training_loop(model, opt, 4000, loss_function, verbose=True, extract_loss_info=extract_info, scheduler=scheduler, loading_bar=True)

# draw results
plt.figure(figsize=(14, 7))
plt.plot(losses, label="Loss")
plt.title("Loss of model")
plt.savefig(f"../demo/losses_S3.png")
plt.clf()

# draw infos
infos = list(zip(*infos))
for i,label in enumerate(["Irreducability", "Relations", "unitary", "urr + rel + uni"]):
    plt.plot(infos[i], label=label)
plt.title("Seperate losses - S3 -> U2")
plt.legend()
plt.savefig("../demo/seperate_S3.png")
