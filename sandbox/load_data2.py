# %% Packages
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

# %% Show slices
fig, axs = plt.subplots(3,1,figsize=(10,16))
axs[0].imshow(Image.open('/work3/soeba/HALOS/Data/Images/train3/hr/m00_0006.jpg'),cmap='gray')
axs[0].axis('off')
axs[1].imshow(Image.open('/work3/soeba/HALOS/Data/Images/train2/hr/m00_0006.jpg'),cmap='gray')
axs[1].axis('off')
axs[2].imshow(Image.open('/work3/soeba/HALOS/Data/Images/train/hr/m00_0006.jpg'),cmap='gray')
axs[2].axis('off')
plt.show()
# %%
