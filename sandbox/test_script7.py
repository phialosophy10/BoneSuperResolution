# %% Packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

# %% Load mask volume
path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_002/mask/volume/f_002.nii"
vol = nib.load(path)
vol = np.array(vol.dataobj)

# %% Floodfill algorithm on selected slice
im = vol[:,:,600]

kernel = np.ones((3, 3), np.uint8)
im = cv2.dilate(im, kernel, iterations=3)
_, im_th = cv2.threshold(im, 60, 255, cv2.THRESH_BINARY)
im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
f_mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, f_mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im = im_th | im_floodfill_inv
im[im == 255] = 1

# %% Plot mask of selected slice
plt.figure()
plt.imshow(im,cmap='gray')
plt.show()
# %%
