# %% Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# %% Load a volume

path = '/work3/soeba/HALOS/Data/'

vol_m = nib.load(path+'microCT/femur01_mod.nii')
vol_c = nib.load(path+'clinicalCT/resliced_femur01_paper_linear.nii')

vol_m = np.array(vol_m.dataobj)
vol_c = np.array(vol_c.dataobj)

# %% Select ROI

vol_m = vol_m[:,:,200:1175]
vol_c = vol_c[:,:,200:1175]

# %% Normalize
m_min = np.min(vol_m)
m_max = np.max(vol_m)

c_min = np.min(vol_c)
c_max = np.max(vol_c)

vol_m_norm = (vol_m-m_min)/(m_max-m_min)
vol_c_norm = (vol_c-c_min)/(c_max-c_min)

# %% Show slices side-by-side

slice_id = 350

fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].imshow(vol_c_norm[:,:,slice_id],cmap='gray')
axs[0].axis('off')
axs[1].imshow(vol_m_norm[:,:,slice_id],cmap='gray')
axs[1].axis('off')
plt.show()

# %% Save slices as images

save_path = '/work3/soeba/HALOS/Data/Images/example_slices/'

im_m = Image.fromarray(np.uint8(vol_m_norm[:,:,slice_id] * 255)).convert("L")
im_m.save(save_path+'m01_'+str(slice_id).zfill(4)+'.jpg')
im_c = Image.fromarray(np.uint8(vol_c_norm[:,:,slice_id] * 255)).convert("L")
im_c.save(save_path+'c01_'+str(slice_id).zfill(4)+'.jpg')
# %%
