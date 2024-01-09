# %% Packages
import torch
import torch.nn as nn
import numpy as np

# %% Test

a = torch.randn((1, 129, 129))

m = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

output = m(a)

print(output.shape)
# %%
