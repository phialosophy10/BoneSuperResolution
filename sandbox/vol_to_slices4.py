# %% Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# %% Load a volume
path = '/work3/soeba/HALOS/Data/'

vol_m = nib.load(path+'microCT/femur74_mod.nii') #01, 15, 21, 74
vol_m = np.array(vol_m.dataobj)

# if loading vol_c directly
vol_c = nib.load(path+'clinicalCT/resliced_femur74_paper_linear.nii') #01, 15, 21, 74
vol_c = np.array(vol_c.dataobj)

# If creating vol_c from vol_m
# vol_tmp = np.copy(vol_m)
# vol_tmp = vol_tmp[::4,::4]
# vol_c = np.repeat(np.repeat(vol_tmp,4,axis=1),4,axis=0)

# %% Select ROI
# femur01
# vol_m = vol_m[:,:,200:1175]
# vol_c = vol_c[:,:,200:1175]
# block_x = 4
# block_y = 5

# femur15a
# vol_m = vol_m[:,:,150:675]
# vol_m = vol_m.transpose((1, 0, 2))
# vol_c = vol_c[:,:,150:675]
# vol_c = vol_c.transpose((1, 0, 2))
# block_x = 4
# block_y = 6

# femur15b
# vol_m = vol_m[:,:,676:1200]
# vol_m = vol_m.transpose((1, 0, 2))
# vol_c = vol_c[:,:,676:1200]
# vol_c = vol_c.transpose((1, 0, 2))
# block_x = 4
# block_y = 6

# femur21a
# vol_m = vol_m[:,:,125:675]
# vol_c = vol_c[:,:,125:675]
# block_x = 5
# block_y = 6

# femur21b
# vol_m = vol_m[:,:,676:1150]
# vol_c = vol_c[:,:,676:1150]
# block_x = 5
# block_y = 6

# femur74a
# vol_m = vol_m[:,:,150:675]
# vol_c = vol_c[:,:,150:675]
# block_x = 4
# block_y = 6

# femur74b
vol_m = vol_m[:,:,676:1250]
vol_c = vol_c[:,:,676:1250]
block_x = 4
block_y = 6

# %% Histogram of intensities

# plt.hist(vol_m.ravel(), bins=40, density=True)
# plt.show()

# %% Normalize
m_min = np.min(vol_m)
m_max = np.max(vol_m)
# m_mean = np.mean(vol_m)
# m_std = np.std(vol_m)

c_min = np.min(vol_c)
c_max = np.max(vol_c)
# c_mean = np.mean(vol_c)
# c_std = np.std(vol_c)

vol_m_norm = (vol_m-m_min)/(m_max-m_min)
vol_c_norm = (vol_c-c_min)/(c_max-c_min)

# %% Plot images 
# slice_id = 300
# fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9,7))
# axs[0].imshow(vol_m[:,:,slice_id],cmap='gray')
# axs[1].imshow(vol_c[:,:,slice_id],cmap='gray')
# plt.show()

# %% Make binary
# vol_m[vol_m<=65] = 0
# vol_m[vol_m>65] = 1

# vol_c[vol_c<=65] = 0
# vol_c[vol_c>65] = 1

# %% Plot slice with colorbar
# plt.imshow(vol_c_norm[:,:,300])
# plt.colorbar()
# plt.show()

# %% Plot subslices 
# slice_id = 300
# fig, axs = plt.subplots(block_x, block_y, sharey=True, figsize=(9,9))
# for i in range(block_x):
#     for j in range(block_y):
#         axs[i,j].imshow(Image.fromarray(np.uint8(vol_m_norm[i*128:(i+1)*128,j*128:(j+1)*128,slice_id] * 255)).convert('L'),cmap='gray')
#         axs[i,j].set_title(f'{np.mean(vol_m_norm[i*128:(i+1)*128,j*128:(j+1)*128,slice_id]):.4f}')
# plt.show()

# %% Save volume as images slices

n_ims = vol_m.shape[2]
split_ind = int(n_ims/5)
inds = list(range(n_ims))
random.shuffle(inds)
test_inds = inds[:split_ind]
train_inds = inds[split_ind:]

count = 0
for i in range(len(test_inds)):
    for j in range(block_x):
        for k in range(block_y):
            if np.mean(vol_m_norm[j*128:(j+1)*128,k*128:(k+1)*128,test_inds[i]]) > 0.12: #0.15, 0.12, 0.18, 0.12
                im_m = Image.fromarray(np.uint8(vol_m_norm[j*128:(j+1)*128,k*128:(k+1)*128,test_inds[i]] * 255)).convert("L")
                im_m.save(path+'Images/test_low-res/linear/hr/m74b_'+str(count).zfill(4)+'.jpg') ##01a, 15a, 15b, 21a, 21b, 74a, 74b
                im_c = Image.fromarray(np.uint8(vol_c_norm[j*128:(j+1)*128,k*128:(k+1)*128,test_inds[i]] * 255)).convert("L")
                im_c.save(path+'Images/test_low-res/linear/lr/c74b_'+str(count).zfill(4)+'.jpg') ##01a, 15a, 15b, 21a, 21b, 74a, 74b
                count += 1

count = 0
for i in range(len(train_inds)):
    for j in range(block_x):
        for k in range(block_y):
            if np.mean(vol_m_norm[j*128:(j+1)*128,k*128:(k+1)*128,train_inds[i]]) > 0.12: #0.15, 0.12, 0.18, 0.12
                im_m = Image.fromarray(np.uint8(vol_m_norm[j*128:(j+1)*128,k*128:(k+1)*128,train_inds[i]] * 255)).convert("L")
                im_m.save(path+'Images/train_low-res/linear/hr/m74b_'+str(count).zfill(4)+'.jpg') ##01a, 15a, 15b, 21a, 21b, 74a, 74b
                im_c = Image.fromarray(np.uint8(vol_c_norm[j*128:(j+1)*128,k*128:(k+1)*128,train_inds[i]] * 255)).convert("L")
                im_c.save(path+'Images/train_low-res/linear/lr/c74b_'+str(count).zfill(4)+'.jpg') ##01a, 15a, 15b, 21a, 21b, 74a, 74b
                count += 1

# %%
