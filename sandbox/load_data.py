# %% Packages
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# %% Load volumes

vol_c = nib.load('/work3/soeba/HALOS/Data/clinicalCT/resliced_clinical_femur21.nii.gz')
vol_m = nib.load('/work3/soeba/HALOS/Data/microCT/femur21_mod.nii')

# %% Print sizes
im_c = np.array(vol_c.dataobj)
#im_c = im_c.transpose((1, 0, 2))
im_m = np.array(vol_m.dataobj)
#im_m = im_m.transpose((1, 0, 2))

print(im_c.shape)
print(im_m.shape)

x_dim = (im_c.shape[0]-515)/2
y_dim = (im_c.shape[1]-714)/2

im_c = im_c[x_dim:x_dim+515,y_dim:y_dim+714,:]

# %% Show slices
fig, axs = plt.subplots(5,2,figsize=(10,16))
axs[0,0].imshow(im_c[:,:,200],cmap='gray')
axs[0,0].axis('off')
axs[0,1].imshow(im_m[:,:,200],cmap='gray')
axs[0,1].axis('off')
axs[1,0].imshow(im_c[:,:,450],cmap='gray')
axs[1,0].axis('off')
axs[1,1].imshow(im_m[:,:,450],cmap='gray')
axs[1,1].axis('off')
axs[2,0].imshow(im_c[:,:,700],cmap='gray')
axs[2,0].axis('off')
axs[2,1].imshow(im_m[:,:,700],cmap='gray')
axs[2,1].axis('off')
axs[3,0].imshow(im_c[:,:,950],cmap='gray')
axs[3,0].axis('off')
axs[3,1].imshow(im_m[:,:,950],cmap='gray')
axs[3,1].axis('off')
axs[4,0].imshow(im_c[:,:,1150],cmap='gray')
axs[4,0].axis('off')
axs[4,1].imshow(im_m[:,:,1150],cmap='gray')
axs[4,1].axis('off')
plt.show()
# %%
