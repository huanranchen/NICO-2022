import torch.nn as nn
import torch

a=nn.Linear(3,2)
for name, param in a.named_parameters():
    param.data = torch.zeros(3)

for name, param in a.named_parameters():
    print(param, param.data)