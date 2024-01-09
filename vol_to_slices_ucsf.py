## Loads .nii-volume of the mask, the micro CT and the clinical CT and creates .npy files of
## the whole volume, the slices and 128x128 patches.
## We use the masks to discard empty patches.

# %% Packages
import nibabel as nib
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

# %% Load a volume
root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/"
x_dim, y_dim = 128, 128                 # Set dimension of image patches
num_pix_for_patch = 800                 # Threshold for determining if patch is empty (or almost empty)

bone = "02-01"
if len(sys.argv) > 1:
    bone = sys.argv[1]

# %% Make directories
root_path += "SP" + bone + "/"
res_list = ["mask", "XCT", "mct"]
dir_list = []
for res in res_list:
    dir_list.append(root_path + res + "/vol/")
    dir_list.append(root_path + res + "/slices/")
    dir_list.append(root_path + res + "/patches/")
    if res != "mask":
        dir_list.append(root_path + res + "/slices/images/")
        dir_list.append(root_path + res + "/patches/images/")

for direc in dir_list:
    if not os.path.exists(direc):
        os.makedirs(direc)

# %% Process the three volumes and save as slices and patches
res = "mct" #must be run for all three modalities: "mask", "XCT", "mct"
path = root_path + "dicom/" + res + "/"
vol = nib.load(path + "SP" + bone + "_" + res + ".nii")
vol = np.array(vol.dataobj)

block_x = vol.shape[0]//x_dim           # Calculating the number of patches to divide the volume into in the x-direction
block_y = vol.shape[1]//y_dim           # ...and in the y-direction

# Make boolean array for registry of non-empty patches
full_patches = np.full((block_x, block_y, vol.shape[2]), False)

## Process mask volume
for i in range(vol.shape[2]): #[50,150,250,350,450,550,650,750,850,950,1050,1150,1250,1350,1450,1550]: #[200,400,600,800,1000,1200,1400]:
    im = vol[:,:,i]
    im[im == 255] = 1
    
    # Save slice as .npy and .png
    np.save(root_path + res + "/slices/SP" + bone + "_" + str(i).zfill(4), im)
    
    # Split into patches and save
    for j in range(block_x): 
        for k in range(block_y): 
            patch_no = j*block_y+k
            if np.sum(im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]) > num_pix_for_patch:
                full_patches[j,k,i] = True
                patch = im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                np.save(root_path + res + "/patches/SP" + bone + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch)

np.save(root_path + res + "/vol/SP" + bone, vol)

for res in ["XCT", "mct"]:
    path = root_path + "dicom/" + res + "/"
    vol = nib.load(path + "SP" + bone + "_" + res + ".nii")
    vol = np.array(vol.dataobj)

    # Save volume as .npy
    np.save(root_path + res + "/vol/SP" + bone, vol)
    
    for i in range(vol.shape[2]): #[50,150,250,350,450,550,650,750,850,950,1050,1150,1250,1350,1450,1550]: #[200,400,600,800,1000,1200,1400]:            #vol.shape[2]
        im = vol[:,:,i].astype(np.float32) 
        
        # Normalize slice
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        
        # Save slice as .npy and .png
        np.save(root_path + res + "/slices/SP" + bone + "_" + str(i).zfill(4), im)
        im_uint8 = np.copy(im)
        im_uint8 = np.uint8(im_uint8 * 255)
        cv2.imwrite(root_path + res + "/slices/images/SP" + bone + "_" + str(i).zfill(4) + ".png", im_uint8)
        
        # Split into patches and save
        for j in range(block_x): 
            for k in range(block_y): 
                patch_no = j*block_y+k
                if full_patches[j,k,i]:
                    patch = im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                    np.save(root_path + res + "/patches/SP" + bone + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch)
                    patch_uint8 = im_uint8[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                    cv2.imwrite(root_path + res + "/patches/images/SP" + bone + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2) + ".png", patch_uint8)
# %%
