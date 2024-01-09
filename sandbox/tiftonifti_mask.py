# %% Packages
import sys
from skimage import io
import numpy as np
import nibabel as nib

# %% Load .tif stack
femur_no = "001"            # Which femur to process
if len(sys.argv) > 1:       # or, input from command line
    femur_no = sys.argv[1]

path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_"+femur_no+"/mask/volume/"
vol_clinical = io.imread(path+"f_"+femur_no+"_clinical"+".tif")
vol_micro = io.imread(path+"f_"+femur_no+"_micro"+".tif")

# %% Transpose volume array, so that it becomes (x, y, z)
vol_clinical = vol_clinical.transpose((2, 1, 0))
vol_micro = vol_micro.transpose((2, 1, 0))

# %% Find intersection of volumes
idx_micro = (vol_micro == 0)
vol_clinical[idx_micro] = 0

# %% Simple header for volume
origin = np.array([0,0,0])  # Use [0,0,0] if nothing else is known from the scanner
affine = np.zeros((4,4))
affine[0:3,3] = origin
affine[0,0] = 0.0573016 # spacing in x [mm]
affine[1,1] = 0.0573016 # spacing in y [mm]
affine[2,2] = 0.0573016 # spacing in z [mm]

# %% Save volume
niiVol = nib.Nifti1Image(vol_clinical, affine)
nib.save(niiVol, path+"f_"+femur_no+".nii")
# %%
