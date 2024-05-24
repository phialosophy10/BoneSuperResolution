# %% Packages
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import flood_fill
import matplotlib.pyplot as plt
%matplotlib tk

# %% Loading image and floodfilling
# Read image
im_in = np.load("/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/DTU_data/femur_001/clinical/slices/f_001_0800.npy")

# Threshold.
im_th = im_in > threshold_otsu(im_in)

# Floodfill from point (0, 0).
im_ff = flood_fill(im_th, (0, 0), 1)

# Invert floodfilled image
im_ff_inv = ~im_ff

# Combine the two images to get the foreground.
im_out = im_th | im_ff_inv

# %% Displaying images
fig, axs = plt.subplots(2,2)
axs[0,0].imshow(im_in,cmap='gray')
axs[0,0].set_title('Original image')
axs[0,1].imshow(im_th,cmap='gray')
axs[0,1].set_title('Thresholded image')
axs[1,0].imshow(im_ff_inv,cmap='gray')
axs[1,0].set_title('Inverted floodfilled image')
axs[1,1].imshow(im_out,cmap='gray')
axs[1,1].set_title('Mask image')
# %%
def test_func(var):
    return var+5, var+10

one_val = test_func(5)
print(one_val)
# %%
