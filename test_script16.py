# %% Packages 
import numpy as np
import matplotlib.pyplot as plt

# %% Load volume
vol = np.load("/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_001/micro/volume/f_001_centered.npy")

# %% Choose number of slices to plot
cols, rows = 10, 10
slice_ids = list(range(0, vol.shape[2], vol.shape[2] // (cols*rows)))

# %% Make plot of a bunch of slices
fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(15, 15),
                        layout="constrained")
count = 0
for row in range(cols):
    for col in range(rows):
        axs[row, col].imshow(vol[:,:,slice_ids[count]],cmap='gray')
        axs[row, col].axis("off")
        count += 1
fig.suptitle('slices')
plt.show()
# %%
