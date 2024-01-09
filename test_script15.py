# %% Packages
import numpy as np
import matplotlib.pyplot as plt

# %% Load volume and display shape
femur_no = '001'
vol = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur_no + '/clinical/volume/f_' + femur_no + '_centered.npy')
#mask = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur_no + '/mask/volume/f_' + femur_no + '.npy')
#print(vol.shape)

# %% Show slice
slice_id = 1400
im = vol[:,:,slice_id]
#im_mask = mask[:,:,slice_id]
#x_center = np.argmax(np.sum(im_mask,axis=0))
#y_center = np.argmax(np.sum(im_mask,axis=1))
plt.imshow(im,cmap='gray')
#plt.plot(x_center,y_center, 'ro')
plt.show()

# %%
