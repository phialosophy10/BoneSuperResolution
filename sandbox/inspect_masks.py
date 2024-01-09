# %% Packages
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

# %% Set data paths
mask_paths = []
hr_paths = []
femur_no = ["01", "15", "21", "74"]
dataset_path = "/work3/soeba/HALOS/Data/Images"
for i in range(len(femur_no)):
    mask_paths += sorted(glob.glob(dataset_path + "/masks/" + femur_no[i] + "*.*"))
    hr_paths += sorted(glob.glob(dataset_path + "/micro/" + femur_no[i] + "*.*"))

# %% Show a bunch of images to inspect the masks
inds = random.sample(list(range(len(mask_paths))), 100)
#masks = random.sample(mask_paths, 100)
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(5):
        axs[i,j*2].imshow(np.load(mask_paths[inds[i*10+j]]),cmap='gray')
        axs[i,j*2].axis('off')
        axs[i,j*2+1].imshow(np.load(hr_paths[inds[i*10+j]]),cmap='gray')
        axs[i,j*2+1].axis('off')
plt.suptitle('Random sample of masks')
plt.show()
# %%
