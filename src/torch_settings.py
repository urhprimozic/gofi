import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cpu > cuda za naš trashass use case
device = torch.device("cpu")
print("Using device:", device)