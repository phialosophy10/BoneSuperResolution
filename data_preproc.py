## Script that performs the following for both DTU and UCSF datasets:
## (1) Open registered LR .nii-file and save as TIFF and uint16 numpy array
## (2) Open HR TIFF-file and save as uint16 numpy array
## (3) Create synthetic LR data and save as TIFF and uint16 numpy array
## (4) Create masks and save as boolean numpy array

# %% Import packages
import numpy as np
import nibabel as nib
import pydicom as dcm
import tifffile
import os
import glob

import cv2
from skimage.transform import downscale_local_mean, resize
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import flood_fill

# %% directories and functions for getting file path, creating synthetic data and creating masks
rootdir = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/SR_proj/data/'
dtu_bones = [b for b in os.listdir(rootdir + 'DTU/') if b.startswith('f_')]
dtu_test = ['f_002', 'f_086', 'f_138']
ucsf_bones = [b for b in os.listdir(rootdir + 'UCSF/') if b.startswith('SP')]
ucsf_test = ['SP02-01','SP03-01','SP04-01','SP05-01']
npy_dt = np.uint16

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

blurring = True
scale_factor = {'DTU': 4, 'UCSF': 3}
sigma = 1.2
def make_synth(vol,scale=4,blur=True,sigma=1.2):
    '''
    Given a volume and transform params, creates synthetic LR volume.
    
    Parameters
        vol: HR volume (npy-array)
        scale: scaling factor for down-and up-scaling
        blur: whether or not to perform Gaussian blurring
        sigma: kernel for Gaussian blur
    '''  
    orig_shape = vol.shape
    vol = downscale_local_mean(vol, (scale, scale, scale))
    if blur:
        vol = gaussian(vol, sigma=sigma)
    vol = resize(vol, orig_shape, anti_aliasing=False, order=1)
    return vol

kernel = np.ones((3, 3), np.uint8)
def make_mask(vol):
    '''
    Given an LR volume, creates a binary mask as well as a dilated 
    version to use for testing/analysis purposes.
    
    Parameters
        vol: LR volume to create mask from
    '''  
    mask_vol = np.zeros(vol.shape)
    mask_dil_vol = np.zeros(vol.shape)
    vol = vol > threshold_otsu(vol)
    for i in range(vol.shape[2]):
        im_th = vol[:,:,i]
        im_ff = flood_fill(im_th, (0, 0), 1)
        im_ff_inv = ~im_ff
        mask = im_th | im_ff_inv
        mask_vol[:,:,i] = mask
        mask_dil = cv2.erode(mask.astype(np.uint8), kernel, iterations=10)
        mask_dil = cv2.dilate(mask_dil, kernel, iterations=30)
        mask_dil_vol[:,:,i] = mask_dil
    mask_vol = mask_vol.astype(np.bool)
    mask_dil_vol = mask_dil_vol.astype(np.bool)
    return mask_vol, mask_dil_vol

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

# %% (1)+(4) LR conversion and create mask
for bone in ucsf_bones: # dtu_bones + ucsf_bones:
    #(1)
    print(f'Saving LR of bone {bone} as TIFF and NPY')
    nii_path = get_path(bone,'LR',ext='nii')
    nii_vol = nib.load(nii_path)
    vol = np.ascontiguousarray(np.array(nii_vol.dataobj).transpose(2,0,1))
    vol = vol.astype(npy_dt)
    npy_path = get_path(bone,'LR',ext='npy')
    np.save(npy_path,vol)
    tif_path = get_path(bone,'LR',ext='tif')
    tifffile.imwrite(tif_path,vol)
    #(4)
    print(f'Creating masks for bone {bone} and saving as NPY')
    if bone.startswith('f_'):
        if bone in dtu_test:
            mask, mask_dil = make_mask(vol)
            mask_dil_path = get_path(bone,'MS',suf='dilated',ext='npy')
            np.save(mask_dil_path,mask_dil)
        else:
            mask, _ = make_mask(vol)
        mask_path = get_path(bone,'MS',ext='npy')
        np.save(mask_path,mask)
    elif bone.startswith('SP'):
        if bone in ucsf_test:
            mask, mask_dil = make_mask_ucsf(bone)
            mask_dil_path = get_path(bone,'MS',suf='dilated',ext='npy')
            np.save(mask_dil_path,mask_dil)
        else:
            mask, _ = make_mask_ucsf(bone)
        mask_path = get_path(bone,'MS',ext='npy')
        np.save(mask_path,mask)

# %% (2)+(3) HR conversion and create synthetic data
for bone in ucsf_bones: # dtu_bones + ucsf_bones:
    #(2)
    print(f'Saving HR of bone {bone} as TIFF and NPY')
    tif_path = get_path(bone,'HR',ext='tif')
    vol = tifffile.imread(tif_path)
    vol = vol.astype(npy_dt)
    npy_path = get_path(bone,'HR',ext='npy')
    np.save(npy_path,vol)
    #(3)
    print(f'Creating synthetic LR of bone {bone} and saving as TIFF and NPY')
    if bone.startswith('f_'):
        inst = 'DTU'
    elif bone.startswith('SP'):
        inst = 'UCSF'
    vol_sy = make_synth(vol,scale=scale_factor[inst],blur=blurring,sigma=sigma)
    sy_tif_path = get_path(bone,'SY',ext='tif')
    tifffile.imwrite(sy_tif_path,vol_sy)
    sy_npy_path = get_path(bone,'SY',ext='npy')
    np.save(sy_npy_path,vol_sy)