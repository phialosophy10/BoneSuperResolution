# %% Packages
import numpy as np
import matplotlib.pyplot as plt

# %% Set path
femur_no = "001"
res = "micro/"
slices = ["50","150","250","350","450","550","650","750","850","950","1050","1150","1250","1350","1450","1550"]

path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"
path += "femur_" + femur_no + "/"

k = 1
plt.figure(figsize=(15,15))
for slice_no in slices:
    im_path = path + res + "slices/" "f_" + femur_no + "_" + slice_no.zfill(4) + ".npy"
    mask_path = path + "mask/slices/f_" + femur_no + "_" + slice_no.zfill(4) + ".npy"

    #  Load slice and mask
    im = np.load(im_path)
    mask = np.load(mask_path)

    # Mask image with mask
    idx = (mask == 1)
    masked_im = np.zeros(mask.shape)
    masked_im[idx] = im[idx]

    #  Plot masked image
    plt.subplot(4,4,k)
    plt.imshow(mask,cmap='gray')
    plt.title(f'f_{femur_no} (slice {slice_no})')
    k += 1
plt.show()
# %%
