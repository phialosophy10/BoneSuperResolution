# %% Packages
import nibabel as nib
import numpy as np
import porespy as ps
ps.visualization.set_mpl_style()

import matplotlib.pyplot as plt

# %% Load a volume
path = "/work3/soeba/HALOS/Data/"

femur_no = "74" #"01", "15", "21", "74"

part = "b" #"a", "b"

vol_m = nib.load(path+"microCT/femur" + femur_no + "_mod.nii") #01, 15, 21, 74
vol_m = np.array(vol_m.dataobj)

vol_c_h = nib.load(path+"clinicalCT/resliced_femur" + femur_no + ".nii.gz") #01, 15, 21, 74
vol_c_h = np.array(vol_c_h.dataobj)

vol_c_l = nib.load(path+"clinicalCT/resliced_femur" + femur_no + "_paper_linear.nii") #01, 15, 21, 74
vol_c_l = np.array(vol_c_l.dataobj)

vol_c_n = nib.load(path+"clinicalCT/resliced_femur" + femur_no + "_paper_nn.nii") #01, 15, 21, 74
vol_c_n = np.array(vol_c_n.dataobj)

# %% Select ROI
if femur_no == "01":
    vol_m = vol_m[:,:,200:1175]
    vol_c_h = vol_c_h[:,:,200:1175]
    vol_c_l = vol_c_l[:,:,200:1175]
    vol_c_n = vol_c_n[:,:,200:1175]

    block_x = 4
    block_y = 5

elif femur_no == "15":
    if part == "a":
        vol_m = vol_m[:,:,150:675]
        vol_m = vol_m.transpose((1, 0, 2))
        vol_c_h = vol_c_h[:,:,150:675]
        vol_c_h = vol_c_h.transpose((1, 0, 2))
        vol_c_l = vol_c_l[:,:,150:675]
        vol_c_l = vol_c_l.transpose((1, 0, 2))
        vol_c_n = vol_c_n[:,:,150:675]
        vol_c_n = vol_c_n.transpose((1, 0, 2))

    elif part == "b":
        vol_m = vol_m[:,:,676:1200]
        vol_m = vol_m.transpose((1, 0, 2))
        vol_c_h = vol_c_h[:,:,676:1200]
        vol_c_h = vol_c_h.transpose((1, 0, 2))
        vol_c_l = vol_c_l[:,:,676:1200]
        vol_c_l = vol_c_l.transpose((1, 0, 2))
        vol_c_n = vol_c_n[:,:,676:1200]
        vol_c_n = vol_c_n.transpose((1, 0, 2))

    block_x = 4
    block_y = 6

elif femur_no == "21":
    if part == "a":
        vol_m = vol_m[:,:,125:675]
        vol_c_h = vol_c_h[:,:,125:675]
        vol_c_l = vol_c_l[:,:,125:675]
        vol_c_n = vol_c_n[:,:,125:675]

    elif part == "b":
        vol_m = vol_m[:,:,676:1150]
        vol_c_h = vol_c_h[:,:,676:1150]
        vol_c_l = vol_c_l[:,:,676:1150]
        vol_c_n = vol_c_n[:,:,676:1150]

    block_x = 5
    block_y = 6

elif femur_no == "74":
    if part == "a":
        vol_m = vol_m[:,:,150:675]
        vol_c_h = vol_c_h[:,:,150:675]
        vol_c_l = vol_c_l[:,:,150:675]
        vol_c_n = vol_c_n[:,:,150:675]

    elif part == "b":
        vol_m = vol_m[:,:,676:1250]
        vol_c_h = vol_c_h[:,:,676:1250]
        vol_c_l = vol_c_l[:,:,676:1250]
        vol_c_n = vol_c_n[:,:,676:1250]

    block_x = 4
    block_y = 6

# %% Normalize
m_min = np.min(vol_m)
m_max = np.max(vol_m)

c_h_min = np.min(vol_c_h)
c_h_max = np.max(vol_c_h)

c_l_min = np.min(vol_c_l)
c_l_max = np.max(vol_c_l)

c_n_min = np.min(vol_c_n)
c_n_max = np.max(vol_c_n)

vol_m_norm = (vol_m-m_min)/(m_max-m_min)
vol_c_h_norm = (vol_c_h-c_h_min)/(c_h_max-c_h_min)
vol_c_l_norm = (vol_c_l-c_l_min)/(c_l_max-c_l_min)
vol_c_n_norm = (vol_c_n-c_n_min)/(c_n_max-c_n_min)

# %% Save volume as images slices

if femur_no == "01":
    thresh = 0.15
elif femur_no == "15":
    thresh = 0.12
elif femur_no == "21":
    thresh = 0.18
elif femur_no == "74":
    thresh = 0.12

# %%
count = 0
for i in range(vol_m.shape[2]):
    for j in range(block_x):
        for k in range(block_y):
            if np.mean(vol_m_norm[j*128:(j+1)*128,k*128:(k+1)*128,i]) > thresh:
                im_m = np.uint8(vol_m_norm[j*128:(j+1)*128,k*128:(k+1)*128,i] * 255)
                np.save(path + "Images/micro/" + femur_no + part + "_" + str(count).zfill(4), im_m)
                im_bin = np.copy(im_m)
                im_bin[im_bin <= 102] = 0  #threshold for segmenting bone/not-bone: 102
                im_bin[im_bin > 102] = 1  #threshold for segmenting bone/not-bone: 102
                thick = ps.filters.local_thickness(im_bin, mode='dt')
                np.save(path + "Images/thick/" + femur_no + part + "_" + str(count).zfill(4), thick)
                im_c_h = np.uint8(vol_c_h_norm[j*128:(j+1)*128,k*128:(k+1)*128,i] * 255)
                np.save(path+"Images/clinical/hi-res/" + femur_no + part + "_" + str(count).zfill(4), im_c_h)
                im_c_l = np.uint8(vol_c_l_norm[j*128:(j+1)*128,k*128:(k+1)*128,i] * 255)
                np.save(path+"Images/clinical/low-res/linear/" + femur_no + part + "_" + str(count).zfill(4), im_c_l)
                im_c_n = np.uint8(vol_c_n_norm[j*128:(j+1)*128,k*128:(k+1)*128,i] * 255)
                np.save(path+"Images/clinical/low-res/nn/" + femur_no + part + "_" + str(count).zfill(4), im_c_n)
                count += 1

# %%
