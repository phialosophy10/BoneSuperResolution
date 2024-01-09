# %% Packages
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

# %% Parameters and initialization
x_dim, y_dim = 128, 128
femur_no = "01" # "01", "15", "21", "74"
dataset_path = "/work3/soeba/HALOS/Data/Images"

if femur_no == "01":
    block_x = 4
    block_y = 5
    n_slices = 975
elif femur_no == "15":
    block_x = 4
    block_y = 6
    n_slices = 1050
elif femur_no == "21":
    block_x = 5
    block_y = 6
    n_slices = 1025
elif femur_no == "74":
    block_x = 4
    block_y = 6
    n_slices = 1100

vol_hr = np.zeros((block_x*x_dim,block_y*y_dim,n_slices),dtype=np.uint8)
vol_lr = np.zeros((block_x*x_dim,block_y*y_dim,n_slices),dtype=np.uint8)
#vol_sr = np.zeros((block_x*x_dim,block_y*y_dim,n_slices),dtype=np.uint8)

# %% Load and stitch patches

for i in range(800, 801): #n_slices
    hr = sorted(glob.glob(dataset_path + '/micro/' + femur_no + '?_' + str(i).zfill(4) + "*.*"))
    lr = sorted(glob.glob(dataset_path + '/clinical/low-res/linear/' + femur_no + '?_' + str(i).zfill(4) + "*.*"))
   # sr = sorted(glob.glob(dataset_path + '/SR/low-res/linear/' + femur_no + '?_' + str(i).zfill(4) + "*.*"))
    for j in range(len(hr)):
        patch_no = int(hr[j][-6:-4])
        patch_x_ind = patch_no // block_y
        patch_y_ind = patch_no % block_y
        
        im_hr = np.load(hr[j])
        im_lr = np.load(lr[j])
       # im_sr = np.load(sr[j])

        vol_hr[patch_x_ind*x_dim:(patch_x_ind+1)*x_dim,patch_y_ind*y_dim:(patch_y_ind+1)*y_dim,i] = im_hr
        vol_lr[patch_x_ind*x_dim:(patch_x_ind+1)*x_dim,patch_y_ind*y_dim:(patch_y_ind+1)*y_dim,i] = im_lr
       # vol_sr[patch_x_ind*x_dim:(patch_x_ind+1)*x_dim,patch_y_ind*y_dim:(patch_y_ind+1)*y_dim,i] = im_sr

# %% Plot examples

plt.figure()
plt.imshow(vol_hr[:,:,800],cmap='gray')
plt.title(f'Femur no: ' + femur_no + f' (slice no. {i})')
plt.show()

plt.figure()
plt.imshow(vol_lr[:,:,800],cmap='gray')
plt.title(f'Femur no: ' + femur_no + f' (slice no. {i})')
plt.show()

# plt.figure()
# plt.imshow(vol_sr[:,:,800],cmap='gray')
# plt.title(f'Femur no: ' + femur_no + f' (slice no. {i})')
# plt.show()

# %% Save patches as images
np.save('/work3/soeba/HALOS/Data/npy_vols/vol' + femur_no + '_hr', vol_hr)
np.save('/work3/soeba/HALOS/Data/npy_vols/vol' + femur_no + '_lr', vol_lr)
np.save('/work3/soeba/HALOS/Data/npy_vols/vol' + femur_no + '_sr', vol_sr)