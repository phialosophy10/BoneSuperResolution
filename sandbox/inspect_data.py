# %% Import packages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# %% Load slice
dataPath = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/raw_data_extern/Holmens kirke NM538-13 X1 [2022-09-20 14.44.40]/tiff_stack/Holmens kirke NM538B X10582.tif"
im = np.asarray(Image.open(dataPath))

# %% Normalize slice
im_min = np.min(im)
im_max = np.max(im)

im_norm = (im-im_min)/(im_max-im_min)
im_norm = im_norm*255

# %% Discard pixels below 50 and above 100
im_norm = im_norm - 80
im_norm[im_norm<0] = 0
im_norm[im_norm>10] = 0
im_norm = im_norm * 25

# %% Normalize slice again
# im_min = np.min(im_norm)
# im_max = np.max(im_norm)

# im_norm = (im_norm-im_min)/(im_max-im_min)
# im_norm = im_norm*255

# %% Show slice
plt.hist(im_norm.ravel(), bins=range(256), fc='k', ec='k')
# %%
plt.imshow(im_norm,cmap='gray')
# %%
