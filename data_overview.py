# %% Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# %% Helper function
def get_file_path(folder, res, ext='npy'):
    '''
    Given folder, resolution and extension returns full file name.
    
    Parameters
        folder: either 'femur_xxx' or 'SP_xx_xx'
        res: either 'LR', 'HR', 'SY' or 'MS'
        ext: either 'npy' or some other format
    '''
    if folder.startswith('SP'):
        rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/'
        subdir = {'LR': '/XCT', 'HR': '/mct', 'SY': '/synth', 'MS': '/mask'}
        vol = '/vol/'
        filename = folder
    elif folder.startswith('femur_'):
        rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/'
        subdir = {'LR': '/clinical', 'HR': '/micro', 'SY': '/synth', 'MS': '/mask'}
        vol = '/volume/'
        filename = f'f_{folder[-3:]}'
    else:
        return
    
    # if res == 'LR':
    #     filepath = rootdir + folder + subdir[res] + vol + filename + '_padded' + '.' + ext
    # else:
    #     filepath = rootdir + folder + subdir[res] + vol + filename + '.' + ext
        
    filepath = rootdir + folder + subdir[res] + vol + filename + '.' + ext
    
    return filepath

# %% Get hold of all files
rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis'
ucsf = '/UCSF_data'
dtu_datadirs = [f for f in os.listdir(rootdir) if f.startswith('femur_')]
ucsf_datadirs = [f for f in os.listdir(rootdir + ucsf) if f.startswith('SP')]
res = ['LR', 'HR', 'SY', 'MS']

# %% Check whether we have all files
for folder in dtu_datadirs: # + ucsf_datadirs:
    print(f'{folder:<9}')
    for r in res:
        filepath = get_file_path(folder, r, ext='npy')
        found = os.path.isfile(filepath)
        if found:
            vol = np.load(filepath, mmap_mode='r')
            cont_style = 'F'
            if vol.data.c_contiguous:
                cont_style = 'C'
            print(f'{folder:<9} {r}:{str(vol.shape):<18} {vol.dtype} {cont_style}')
        else:
            print(r, 'not found', end= ', ')
# %%
