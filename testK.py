import torch
from dpt import dpt_tiny


model = dpt_tiny()

inp = torch.rand((10,3,224,224))
out = model.forward_features(inp)
print(inp.shape)
