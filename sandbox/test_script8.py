# %% Packages
import nibabel as nib

# %% Load volume
femur_no = "001"
res = "mask"

path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"
path += "femur_" + femur_no + "/"
path += res + "/"
vol = nib.load(path + "volume/f_" + femur_no + ".nii")
# %%
