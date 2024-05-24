# %% Import packages
import numpy as np
import pydicom as dcm
import cv2
import glob
import matplotlib.pyplot as plt

# %% directoriy and functions
rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/SR_proj/data/'

def get_path(bone, res, suf='', ext='nii'):
    '''
    Given bone, resolution, suffix (optional) and extension, returns full file name.
    
    Parameters
        bone: either 'f_xxx' or 'SP_xx_xx'
        res: either 'LR', 'HR', 'SY' or 'MS'
        suf: e.g. 'dilated'
        ext: e.g. 'npy'
    '''  
    if bone.startswith('f_'):
        inst = 'DTU/'
    elif bone.startswith('SP'):
        inst = 'UCSF/'
    else:
        return
    
    if suf != '':
        suf = '_' + suf
        
    return rootdir + inst + bone + '/' + res + '/' + bone + suf + '.' + ext

kernel = np.ones((3, 3), np.uint8)
def make_mask_ucsf(bone):
    '''
    Given a bone, returns the mask, which is a combination of the trabecular and cortical masks
    '''
    cort_files = sorted(glob.glob(rootdir + 'UCSF/raw_data/' + bone + '/MS/CORT_MASK/' + "*.*"))
    trab_files = sorted(glob.glob(rootdir + 'UCSF/raw_data/' + bone + '/MS/TRAB_MASK/' + "*.*"))
    cort_ex = dcm.dcmread(cort_files[0]).pixel_array
    mask = np.zeros((cort_ex.shape[0],cort_ex.shape[1],len(cort_files)))
    mask_dil = np.zeros(mask.shape)
    for i in range(mask.shape[2]):
        cort_mask = dcm.dcmread(cort_files[i]).pixel_array
        trab_mask = dcm.dcmread(trab_files[i]).pixel_array
        mask_im = np.zeros(cort_mask.shape)
        idxs_cort = np.argwhere(cort_mask > 0)
        idxs_trab = np.argwhere(trab_mask > 0)
        idxs_cort = np.nonzero(cort_mask)
        idxs_trab = np.nonzero(trab_mask)
        mask_im[idxs_cort] = 1
        mask_im[idxs_trab] = 1
        mask[:,:,i] = mask_im
        mask_dil_im = cv2.erode(mask_im.astype(np.uint8), kernel, iterations=10)
        mask_dil_im = cv2.dilate(mask_dil_im, kernel, iterations=30)
        mask_dil[:,:,i] = mask_dil_im
    mask = mask.astype(np.bool)
    mask_dil = mask_dil.astype(np.bool)
    return mask, mask_dil

# %% Make and save mask
bone = "SP02-01"
mask, mask_dil = make_mask_ucsf(bone)
mask_dil_path = get_path(bone,'MS',suf='dilated',ext='npy')
np.save(mask_dil_path,mask_dil)
mask_path = get_path(bone,'MS',ext='npy')
np.save(mask_path,mask)
# %% Print and plot some stuff
print(f'Datatype for mask: {mask.dtype}')
print(f'Datatype for dilated mask: {mask_dil.dtype}')
plt.figure()
plt.imshow(mask[:,:,100],cmap='gray')
plt.show()
plt.figure()
plt.imshow(mask_dil[:,:,100],cmap='gray')
plt.show()
# %%
