## Loads .nii-volume of the mask, the micro CT and the clinical CT and creates .npy files of
## the whole volume, the slices and 128x128 patches.
## We modify the binary segmentation mask, so that it segments inside/outside bone.
## We also use the masks to discard empty patches.

# %% Packages
import nibabel as nib
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

# %% Load a volume
root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"

femur_no = "026"                        # [001,002,013,015,021,026,031,074,075,083,086,138,164,172]
if len(sys.argv) > 1:
    femur_no = sys.argv[1]

root_path += "femur_" + femur_no + "/"

x_dim, y_dim = 128, 128                 # Set dimension of image patches
num_pix_for_patch = 800                 # Threshold for determining if patch is empty (or almost empty)
kernel = np.ones((3, 3), np.uint8)

# %% Process the three volumes and save as slices and patches
res = "mask/"
path = root_path + res
vol = nib.load(path + "volume/f_" + femur_no + ".nii")
vol = np.array(vol.dataobj)

block_x = vol.shape[0]//x_dim           # Calculating the number of patches to divide the volume into in the x-direction
block_y = vol.shape[1]//y_dim           # ...and in the y-direction

# Make boolean array for registry of non-empty patches
full_patches = np.full((block_x, block_y, vol.shape[2]), False)

## Process mask volume
for i in range(vol.shape[2]): #[50,150,250,350,450,550,650,750,850,950,1050,1150,1250,1350,1450,1550]: #[200,400,600,800,1000,1200,1400]:
    im = vol[:,:,i]
    
    # Floodfill algorithm to obtain binary mask of slice (inside bone/outside bone)
    for _ in range(7):
        im = cv2.dilate(im, kernel, iterations=2)
        im = cv2.erode(im, kernel, iterations=1)
    im_floodfill = im.copy()
    h, w = im.shape[:2]
    f_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, f_mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im = im | im_floodfill_inv
    if (i < 100):
        im = cv2.erode(im, kernel, iterations=10)
    elif (100 <= i) and (i < 200):
        im = cv2.erode(im, kernel, iterations=7)
    else:
        im = cv2.erode(im, kernel, iterations=3)
    im[im == 255] = 1
    
    # Save slice as .npy
    np.save(path + "slices/f_" + femur_no + "_" + str(i).zfill(4), im)
    
    # Split into patches and save
    for j in range(block_x): 
        for k in range(block_y): 
            patch_no = j*block_y+k
            if np.sum(im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]) > num_pix_for_patch:
                # full_patches[j,k,i] = True
                patch = im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                np.save(path + "patches/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch)
    
    vol[:,:,i] = im     # Update the slice in the volume with the floodfilled mask

# Save volume as .npy
np.save(path + "volume/f_" + femur_no, vol)

## Process CT volumes
for res in ["clinical/", "micro/"]:
    path = root_path + res
    vol = nib.load(path + "volume/f_" + femur_no + ".nii")
    vol = np.array(vol.dataobj)
    
    block_x = vol.shape[0]//x_dim           # Calculating the number of patches to divide the volume into in the x-direction
    block_y = vol.shape[1]//y_dim           # ...and in the y-direction
    
    # Save volume as .npy
    np.save(path + "volume/f_" + femur_no, vol)

    for i in range(vol.shape[2]): #[50,150,250,350,450,550,650,750,850,950,1050,1150,1250,1350,1450,1550]: #[200,400,600,800,1000,1200,1400]:            #vol.shape[2]
        im = vol[:,:,i]
        
        # Normalize slice
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        
        # Save slice as .npy and .png
        np.save(path + "slices/f_" + femur_no + "_" + str(i).zfill(4), im)
        im_uint8 = np.copy(im)
        im_uint8 = np.uint8(im_uint8 * 255)
        cv2.imwrite(path + "slices/images/f_" + femur_no + "_" + str(i).zfill(4) + ".png", im_uint8)
        
        # Split into patches and save
        for j in range(block_x): 
            for k in range(block_y): 
                patch_no = j*block_y+k
                if full_patches[j,k,i]:
                    patch = im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                    np.save(path + "patches/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch)
                    patch_uint8 = im_uint8[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                    cv2.imwrite(path + "patches/images/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2) + ".png", patch_uint8)
# %%
