# %% Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# %% Load a volume

path = '/work3/soeba/HALOS/Data/'

vol_m = nib.load(path+'microCT/femur01_mod.nii')
#vol_c = nib.load(path+'clinicalCT/resliced_clinical_femur01.nii.gz')

vol_m = np.array(vol_m.dataobj)
#vol_m = vol_m.transpose((1, 0, 2)) # For femur15
#vol_c = np.array(vol_c.dataobj)
#vol_c = vol_c.transpose((1, 0, 2)) # For femur15

# %% Select ROI

vol_m = vol_m[3:,102:614,200:1175]
# vol_tmp = np.copy(vol_m)
# vol_tmp = vol_tmp[::4,::4]
# vol_c = np.repeat(np.repeat(vol_tmp,4,axis=1),4,axis=0)

#vol_c = vol_c[3:,102:614,200:1175]

# vol_m = vol_m[:,:,150:1200]
# vol_c = vol_c[:,:,150:1200]

# vol_m = vol_m[:,:,125:1150]
# vol_c = vol_c[:,:,125:1150]

# vol_m = vol_m[:,:,150:1250]
# vol_c = vol_c[:,:,150:1250]

# %% Normalize
# m_mean = np.mean(vol_m)
# m_std = np.std(vol_m)

# vol_m_norm = (vol_m-m_mean)/m_std
#vol_c_norm = (vol_c-m_mean)/m_std

# %% Plot images 
# slice_id = 500
# fig, axs = plt.subplots(2, 2, sharey=True, figsize=(9,7))
# axs[0,0].imshow(vol_m[:,:,slice_id],cmap='gray')
# axs[0,1].imshow(vol_c[:,:,slice_id],cmap='gray')
# axs[1,0].imshow(vol_m_norm[:,:,slice_id],cmap='gray')
# axs[1,1].imshow(vol_c_norm[:,:,slice_id],cmap='gray')
# plt.show()

# %% Make binary
vol_m[vol_m<=65] = 0
vol_m[vol_m>65] = 1

# vol_c[vol_c<=65] = 0
# vol_c[vol_c>65] = 1

# %% Plot
plt.imshow(vol_m[:,:,500],cmap='gray')
plt.show()

# %% Save volume as images slices

# np.save(path+'Images/m03',vol_m)
# np.save(path+'Images/c03',vol_c)

for i in range(vol_m.shape[2]):
    im_m = vol_m[:,:,i]
    np.save(path+'bone_test/images/binary2/m00_'+str(i).zfill(4),im_m)

# n_ims = vol_m.shape[2]
# split_ind = int(n_ims/5)
# inds = list(range(n_ims))
# random.shuffle(inds)
# test_inds = inds[:split_ind]
# train_inds = inds[split_ind:]

# for i in range(len(test_inds)):
#     im_m = Image.fromarray(vol_m[:,:,test_inds[i]]).convert("L")
#     im_m.save(path+'Images/test2/hr/m00_'+str(test_inds[i]).zfill(4)+'.jpg')
#     im_c = Image.fromarray(vol_c[:,:,test_inds[i]]).convert("L")
#     im_c.save(path+'Images/test2/lr/m00_'+str(test_inds[i]).zfill(4)+'.jpg')

# for i in range(len(train_inds)):
#     im_m = Image.fromarray(vol_m[:,:,train_inds[i]]).convert("L")
#     im_m.save(path+'Images/train2/hr/m00_'+str(train_inds[i]).zfill(4)+'.jpg')
#     im_c = Image.fromarray(vol_c[:,:,train_inds[i]]).convert("L")
#     im_c.save(path+'Images/train2/lr/m00_'+str(train_inds[i]).zfill(4)+'.jpg')

# %%
