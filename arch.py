import torch
from arl_sp import ARL
from arl_sp import Block

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = ARL(Block, [2, 2, 2, 2]).to(device)

for name, param in model.named_parameters():
    if 'factor' in name:
        print(name, param)


