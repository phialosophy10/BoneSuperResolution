# %% Packages
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# %% Load dicoms
femur_no = "172"
# im1 = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_001/micro/slices/f_001_0000.npy')
im2 = pydicom.dcmread('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/DICOM_micro_CT/' + femur_no + '/' + femur_no + '_slice_500.dcm')
im2 = im2.pixel_array.astype(np.int32)

# %% Normalize
# im2_min = np.min(im2)
# im2_max = np.max(im2)
# im2_norm = (im2 - im2_min) / (im2_max - im2_min)

# %% Plot histograms
plt.figure()
#plt.hist(np.ravel(im2_norm),bins='auto')
plt.imshow(im2,cmap='gray')
plt.show()

# %% New array
im1_new = np.zeros(im1.shape)
indx1 = im1 > 0.5
im1_new[indx1] = im1[indx1]

im2_new = np.zeros(im2_norm.shape)
indx2 = im2_norm < 0.5
im2_new[indx2] = im2_norm[indx2]

# %% Plot slice
plt.figure()
plt.imshow(im2_new,cmap='gray')
plt.show()
# %%
