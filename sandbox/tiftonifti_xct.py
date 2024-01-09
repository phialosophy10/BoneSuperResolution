# %% Packages
import sys
from skimage import io
import numpy as np
import nibabel as nib
import cv2

# %% Load .tif stack
path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF data/Sophia_phantom/mask/"
vol = io.imread(path + "T0000126_SEG.tif")

# %% Transpose volume array, so that it becomes (x, y, z)
vol = vol.transpose((1, 2, 0))

# %% Combine, dilate and save (incl. a few slices as images)
idx = (vol == 65019)
vol[idx] = 0
idx_ones = (vol != 0)
vol[idx_ones] = 1

im = vol[:,:,65]
im = np.uint8(im * 255)
cv2.imwrite(path + "slice_65.png", im)
# %% Simple header for volume
origin = np.array([0,0,0])  # Use [0,0,0] if nothing else is known from the scanner
affine = np.zeros((4,4))
affine[0:3,3] = origin
affine[0,0] = 0.06 # spacing in x [mm]
affine[1,1] = 0.06 # spacing in y [mm]
affine[2,2] = 0.06 # spacing in z [mm]

# %% Save volume
niiVol = nib.Nifti1Image(vol, affine)
nib.save(niiVol, path + "T0000126.nii")
# %%
