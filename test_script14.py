# %% Packages
import pydicom

# %% Load DICOM files
dicom_path_single = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF data/SP02-01/dicom/mct/grayscale/D0000818_00000.dcm'
dicom_path_series = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_001/clinical/1_chiara/IM0001.dcm'
d_sin = pydicom.dcmread(dicom_path_single)
d_ser = pydicom.dcmread(dicom_path_series)
# %%
