## Loads a volume and creates .npy files of the whole volume, the slices and 128x128 patches as well as binary masks for slices and patches.
## We perform binary segmentation (bone/not bone) for every slice and save the segmentation masks.
## We also use the masks to discard empty patches

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
num_pix_for_patch = 400                 # Threshold for determining if patch is empty (or almost empty)

for res in ["clinical/", "micro/"]:               # ["clinical/", "micro/"]
    path = root_path + res
    vol = nib.load(path + "volume/f_" + femur_no + ".nii")
    vol = np.array(vol.dataobj)
    vol_mask = np.zeros(vol.shape)
    
    # Save volume as .npy
    np.save(path + "volume/f_" + femur_no, vol)
    
    # Get normalization values from volume
    # vol_min = np.min(vol)
    # vol_max = np.max(vol)
    # if (res == "clinical/"):
    #     bone_thresh = (np.quantile(vol, 0.75) - vol_min) / (vol_max - vol_min) * 255
    
    block_x = vol.shape[0]//x_dim       # Calculating the number of patches to divide the volume into in the x-direction
    block_y = vol.shape[1]//y_dim       # ...and in the y-direction

    # Slices: Select, normalize, segment to make binary mask, save
    for i in [200,400,600,800,1000,1200,1400]:            #vol.shape[2]
        im = vol[:,:,i]
        
        # Normalize slice
        # im = (im-vol_min)/(vol_max-vol_min)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        
        # Save slice as .npy and .png
        np.save(path + "slices/f_" + femur_no + "_" + str(i).zfill(4), im)
        im_uint8 = np.copy(im)
        im_uint8 = np.uint8(im_uint8 * 255)
        cv2.imwrite(path + "slices/images/f_" + femur_no + "_" + str(i).zfill(4) + ".png", im_uint8)
        
        # plt.figure()
        # plt.imshow(im,cmap='gray')
        # plt.title(f'clinical-CT image (slice no. 650)')
        # plt.show()
        
        if(res == "clinical/"):
            # Floodfill algorithm to obtain binary mask of slice (inside bone/outside bone)
            _, im_th = cv2.threshold(im_uint8, np.quantile(im, 0.75), 255, cv2.THRESH_BINARY) 
            im_floodfill = im_th.copy()
            h, w = im_th.shape[:2]
            f_mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, f_mask, (0,0), 255)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            mask = im_th | im_floodfill_inv
            mask[mask == 255] = 1
        
            # plt.figure()
            # plt.imshow(mask,cmap='gray')
            # plt.title(f'clinical-CT mask (slice no. 650)')
            # plt.show()
        
            # Insert slice mask into volume mask
            vol_mask[:,:,i] = mask
        
            # Save mask as .npy
            np.save(root_path + "mask/slices/f_" + femur_no + "_" + str(i).zfill(4), mask)
        
            # Split into patches and save
            for j in range(block_x): 
                for k in range(block_y): 
                    patch_no = j*block_y+k
                    if np.sum(mask[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]) > num_pix_for_patch:
                        patch = im[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                        np.save(path + "patches/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch)
                        patch_uint8 = im_uint8[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                        cv2.imwrite(path + "patches/images/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2) + ".png", patch_uint8)
                        patch_mask = mask[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
                        np.save(root_path + "mask/patches/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch_mask)
                    
    # Save volume mask
    np.save(root_path + "mask/volume/f_" + femur_no, vol_mask)

# %%
