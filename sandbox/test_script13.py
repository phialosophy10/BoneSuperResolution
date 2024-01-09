# %% Packages
import nibabel as nib
import pydicom
import os
from skimage import io

# %% Load file
dicom_path = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/DICOM_micro_CT/dicom.dcm'
ds = pydicom.dcmread(dicom_path)

# %%
