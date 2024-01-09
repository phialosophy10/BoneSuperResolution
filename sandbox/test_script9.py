# %% Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% Load mask
path1 = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_015/mask/slices/f_015_1450.npy"
path2 = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_015/micro/slices/f_015_1450.npy"

mask = np.load(path1)
im = np.load(path2)

# %% Plot image
plt.figure()
plt.imshow(mask, cmap="gray")
plt.show()

# %% Dilate and erode
mask_tmp = np.copy(mask)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
mask_tmp = cv2.dilate(mask_tmp, np.ones((3, 3), np.uint8), iterations=2)
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=1)

# %% Floodfill
im_floodfill = mask_tmp.copy()
h, w = mask_tmp.shape[:2]
f_mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, f_mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
mask_tmp = mask_tmp | im_floodfill_inv

# %% Erode and dilate again
mask_tmp = cv2.erode(mask_tmp, np.ones((3, 3), np.uint8), iterations=10)

# %% Plot mask
plt.figure()
plt.imshow(mask_tmp, cmap="gray")
plt.show()

# %% Mask image
idx = (mask_tmp == 255)
masked_im = np.zeros(mask_tmp.shape)
masked_im[idx] = im[idx]

# %% Plot masked image
plt.figure()
plt.imshow(masked_im, cmap="gray")
plt.show()
# %%
