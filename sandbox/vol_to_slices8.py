## Loads a volume and creates 128x128 pixel slices. It throws away empty slices
## In this script we only save low-res/linear CT images
## Version that uses OpenCV FloodFill to segment bone from background

# %% Packages
import nibabel as nib
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

# %% Load a volume
path = "/work3/soeba/HALOS/Data/"

femur_no = "01" #"01", "15", "21", "74"
part = "a" #"a", "b"

# if len(sys.argv)>1:
#     femur_no = sys.argv[1]
# if len(sys.argv)>2:
#     part = sys.argv[2]

vol_m = nib.load(path+"microCT/femur" + femur_no + "_mod.nii") #01, 15, 21, 74
vol_m = np.array(vol_m.dataobj)

vol_c_l = nib.load(path+"clinicalCT/resliced_femur" + femur_no + "_paper_linear.nii") #01, 15, 21, 74
vol_c_l = np.array(vol_c_l.dataobj)

x_dim, y_dim = 128, 128

# %% Select ROI
if femur_no == "01":
    vol_m = vol_m[:,:,200:1175]
    vol_c_l = vol_c_l[:,:,200:1175]

    block_x = vol_m.shape[0]//x_dim #4
    block_y = vol_m.shape[1]//y_dim #5

elif femur_no == "15":
    if part == "a":
        vol_m = vol_m[:,:,150:675]
        vol_m = vol_m.transpose((1, 0, 2))
        vol_c_l = vol_c_l[:,:,150:675]
        vol_c_l = vol_c_l.transpose((1, 0, 2))

    elif part == "b":
        vol_m = vol_m[:,:,675:1200]
        vol_m = vol_m.transpose((1, 0, 2))
        vol_c_l = vol_c_l[:,:,675:1200]
        vol_c_l = vol_c_l.transpose((1, 0, 2))

    block_x = vol_m.shape[0]//x_dim #4
    block_y = vol_m.shape[1]//y_dim #6

elif femur_no == "21":
    if part == "a":
        vol_m = vol_m[:,:,125:675]
        vol_c_l = vol_c_l[:,:,125:675]

    elif part == "b":
        vol_m = vol_m[:,:,675:1150]
        vol_c_l = vol_c_l[:,:,675:1150]

    block_x = vol_m.shape[0]//x_dim #5
    block_y = vol_m.shape[1]//y_dim #6

elif femur_no == "74":
    if part == "a":
        vol_m = vol_m[:,:,150:675]
        vol_c_l = vol_c_l[:,:,150:675]

    elif part == "b":
        vol_m = vol_m[:,:,675:1250]
        vol_c_l = vol_c_l[:,:,675:1250]

    block_x = vol_m.shape[0]//x_dim #4
    block_y = vol_m.shape[1]//y_dim #6

# %% Normalize clinical CT scan

c_l_min = np.min(vol_c_l)
c_l_max = np.max(vol_c_l)

vol_c_l_norm = (vol_c_l-c_l_min)/(c_l_max-c_l_min)

# %% Save image slices and masks

# Threshold for determining if slice is empty
num_pix_for_slice = 400

for i in range(800,801): #vol_m.shape[2]
    # Select slice
    im_m = vol_m[:,:,i]
    im_c_l = np.uint8(vol_c_l_norm[:,:,i] * 255)
    # Normalize slice
    im_m_min = np.min(im_m)
    im_m_max = np.max(im_m)
    im_m = (im_m-im_m_min)/(im_m_max-im_m_min)
    # Cast as uint8
    im_m = np.uint8(im_m * 255)
    # Threshold image
    _, im_th = cv2.threshold(im_m, 60, 255, cv2.THRESH_BINARY)
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    im_out[im_out == 255] = 1 #convert to binary array
    for j in range(block_x): #block_x
        for k in range(block_y): #block_y
            patch_no = j*block_y+k
            if np.sum(im_out[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]) > num_pix_for_slice:
                patch_m = im_m[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
              #  np.save(path + "Images/micro/" + femur_no + part + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch_m)
                plt.figure()
                plt.imshow(patch_m,cmap='gray')
                plt.title(f'micro-CT patch (patch no.: {patch_no})')
                plt.show()
              #  patch_c_l = im_c_l[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
              #  np.save(path+"Images/clinical/low-res/linear/" + femur_no + part + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch_c_l)
                # plt.figure()
                # plt.imshow(patch_c_l,cmap='gray')
                # plt.title(f'clinical CT patch (patch no.: {patch_no})')
                # plt.show()
                patch_mask = im_out[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim]
              #  np.save(path+"Images/masks/" + femur_no + part + "_" + str(i).zfill(4) + "_" + str(patch_no).zfill(2), patch_mask)
                plt.figure()
                plt.imshow(patch_mask)
                plt.title(f'mask patch (sum of picels.: {np.sum(patch_mask)})')
                plt.show()

# %%
