# %% Import packages
import numpy as np
import nibabel as nib
import tifffile
import os

# %% Get hold of all file paths
rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/data/'
dtu_bones = [f for f in os.listdir(rootdir + 'DTU/') if f.startswith('f_')]
ucsf_bones = [f for f in os.listdir(rootdir + 'UCSF/') if f.startswith('SP')]

def get_file_path(bone, res, ext='tif'):
    '''
    Given bone and resolution returns full file name.
    
    Parameters
        bone: either 'f_xxx' or 'SP_xx_xx'
        res: either 'LR', 'HR', 'SY' or 'MS'
    '''
    
    rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/data/'
    
    if bone.startswith('SP'):
        inst = 'UCSF/'
    elif bone.startswith('f_'):
        inst = 'DTU/'
    else:
        return
        
    filepath = rootdir + inst + bone + '/' + res + '/' + bone + '.' + ext
    
    return filepath

# Dictionary with path to .vgi-file for DTU data
vgi_dict = {
    "001": "femur_01_b 1 [2022-03-09 10.52.18]/femur_01_b 1_recon/femur_01_b 1.vgi",
    "015": "femur_15 1 [2022-02-25 09.16.41]/femur_15 1_reco/femur_15 1.vgi",
    "021": "femur_21 1 [2022-03-10 10.24.02]/femur_21 1_reco/femur_21 1.vgi",
    "074": "femur_74_c 1 [2022-03-29 14.35.22]/femur_74_c 1_reco/femur_74_c 1.vgi",
    "002": "Holmens kirke NM538-13 X1 [2022-09-20 14.44.40]/Holmens kirke NM538B X1_01_recon/Holmens kirke NM538B X1.vgi",
    "026": "Holmens kirke NM538-13 X26 [2022-09-20 12.14.14]/Holmens kirke NM538B X26_01_recon/Holmens kirke NM538B X26.vgi",
    "083": "Holmens kirke NM538-13 X83 [2022-09-20 10.55.15]/Holmens kirke NM538B X83_recon/Holmens kirke NM538B X83.vgi",
    "138": "Holmens kirke NM538-13 X138 [2022-09-20 13.39.27]/Holmens kirke NM538B X138_01_recon/Holmens kirke NM538B X138.vgi",
    "075": "Holmens kirke NM538-31 X75 [2022-09-20 15.44.44]/Holmens kirke NM538-13 X75_01_recon/Holmens kirke NM538-13 X75.vgi",
    "013": "Holmens_kirke_NM538-13_X13_3141proj [2023-08-29 11.16.46]/Holmens_kirke_NM538-13_X13_3141proj_recon/Holmens_kirke_NM538-13_X13_3141proj.vgi",
    "086": "Holmens_kirke_NM538-13_X86 [2023-08-29 10.04.31]/Holmens_kirke_NM538-13_X86_recon/Holmens_kirke_NM538-13_X86.vgi",
    "164": "Holmens_kirke_NM538-13_X164_new [2023-08-29 14.28.14]/Holmens_kirke_NM538-13_X164_new_recon/Holmens_kirke_NM538-13_X164_new.vgi",
    "172": "Holmens_kirke_NM538-13_X172 [2023-09-11 08.48.53]/Holmens_kirke_NM538-13_X172_recon/Holmens_kirke_NM538-13_X172.vgi"
}

# Function to create affine transformation for Nifti-file
def get_affine(bone):
    '''
    Given bone resturns affine tranformation for Nifti-file
    
    Parameters
        bone: either 'femur_xxx' or 'SP_xx_xx'
    '''
    origin = np.array([0,0,0])
    affine = np.zeros((4,4))
    affine[0:3,3] = origin
    
    if bone.startswith('f_'):
        bone_no = bone[-3:]
        # Read voxel size from .vgi-file
        file = open("/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/raw_data_3DIM/" + vgi_dict[bone_no])
        lines = file.readlines()
        res_line = lines[27].split()
        if res_line[0] == 'resolution':
            affine[0,0] = float(res_line[2]) # spacing in x [mm]
            affine[1,1] = float(res_line[3]) # spacing in y [mm]
            affine[2,2] = float(res_line[4]) # spacing in z [mm]
        else:
            print(f'wrong line read from .vgi-file for bone no. {bone_no}')
    elif bone.startswith('SP'):
        affine[0,0] = 0.0245 # spacing in x [mm]
        affine[1,1] = 0.0245 # spacing in y [mm]
        affine[2,2] = 0.0245 # spacing in z [mm]
    else:
        return
    
    return affine

# %% Check whether we have all files
for bone in dtu_bones + ucsf_bones:
    if bone.startswith('f_'):
        inst = 'DTU/'
        res = ['HR']
    elif bone.startswith('SP'):
        inst = 'UCSF/'
        res = ['HR', 'LR']
    
    for r in res:
        filepath = get_file_path(bone,r,ext='tif')
        affine = get_affine(bone)
        print(filepath)
        print(f'voxelspacing: {affine[0,0]} (x), {affine[1,1]} (y), {affine[2,2]} (z)')
        vol = np.ascontiguousarray(tifffile.imread(filepath).transpose(1,2,0))
        niiVol = nib.Nifti1Image(vol, affine)
        if r == 'LR':
            nib.save(niiVol, rootdir + inst + bone + '/' + r + '/' + bone + "_orig.nii")
        else:
            nib.save(niiVol, rootdir + inst + bone + '/' + r + '/' + bone + ".nii")

# %%
