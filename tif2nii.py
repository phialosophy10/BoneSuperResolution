# %% Import packages
import numpy as np
import nibabel as nib
import tifffile
import os

# %% Get hold of all file paths
rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/SR_proj/data/'
dtu_bones = [f for f in os.listdir(rootdir + 'DTU/') if f.startswith('f_')]
ucsf_bones = [f for f in os.listdir(rootdir + 'UCSF/') if f.startswith('SP')]

def get_file_path(bone, res, ext='tif'):
    '''
    Given bone and resolution returns full file name.
    
    Parameters
        bone: either 'f_xxx' or 'SP_xx_xx'
        res: either 'LR', 'HR', 'SY' or 'MS'
    '''    
    if bone.startswith('SP'):
        inst = 'UCSF/'
    elif bone.startswith('f_'):
        inst = 'DTU/'
    else:
        return
        
    filepath = rootdir + inst + bone + '/' + res + '/' + bone + '.' + ext
    
    return filepath

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
        # Read voxel size from .vgi-file
        file = open(rootdir + 'DTU/raw_data/' + bone + '/HR/' + bone + '.vgi')
        lines = file.readlines()
        res_line = lines[27].split()
        if res_line[0] == 'resolution':
            affine[0,0] = float(res_line[2]) # spacing in x [mm]
            affine[1,1] = float(res_line[3]) # spacing in y [mm]
            affine[2,2] = float(res_line[4]) # spacing in z [mm]
        else:
            print(f'wrong line read from .vgi-file for bone {bone}')
    elif bone.startswith('SP'):
        affine[0,0] = 0.0245 # spacing in x [mm]
        affine[1,1] = 0.0245 # spacing in y [mm]
        affine[2,2] = 0.0245 # spacing in z [mm]
    else:
        return
    
    return affine

# %% Create nii-files
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
