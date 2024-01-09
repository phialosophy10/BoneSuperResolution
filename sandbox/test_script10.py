#%% Packages

import numpy as np

#%% Load a volume

vol = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_001/micro/volume/f_001.npy')

#%% Slice volume

im = vol[:,:,650]

#%% Print shape

print(im.shape)
# %%
