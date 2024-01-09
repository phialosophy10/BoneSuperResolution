# %% Packages
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# %% Load volume
femur_no = "001"
res = "mask/" #"micro/" #"clinical/"
path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_" + femur_no + "/" + res + "/volume/f_" + femur_no + ".npy"

vol = np.load(path)

# %% Plot histogram of volume
plt.figure()
plt.hist(np.ravel(vol), bins="auto")
plt.show()

# %% Select slice
slice_no = 200
im = vol[:,:,slice_no]

# %% Plot histogram of slice
plt.figure()
plt.hist(np.ravel(im), bins="auto")
plt.show()

# %% Choose value region of slice
quant = 0.90
idx = (np.quantile(im, quant) < im) #& (im != 0)
mask = np.zeros(im.shape)
mask[idx] = 1

# %% Dilate and erode
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=3)
mask = cv2.erode(mask, kernel, iterations=1)

# %% Plot map of selected region
plt.figure()
plt.imshow(im, cmap="gray")
plt.show()

# %%
