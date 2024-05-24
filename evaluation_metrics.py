## Packages
import numpy as np
import skimage.metrics as skim
from skimage.filters import threshold_otsu
import localthickness as lt

## Performance metrics
def psnr_vol(hr_vol, sr_vol):
    psnr = np.zeros(hr_vol.shape[2])
    for i in range(hr_vol.shape[2]):
        psnr[i] = skim.peak_signal_noise_ratio(hr_vol[:,:,i], sr_vol[:,:,i])
        
    psnr_mean = np.mean(psnr)
    psnr_std = np.std(psnr)
    
    return psnr_mean, psnr_std

def ssim_vol(hr_vol, sr_vol):
    ssim = np.zeros(hr_vol.shape[2])
    for i in range(hr_vol.shape[2]):
        ssim[i] = skim.structural_similarity(hr_vol[:,:,i], sr_vol[:,:,i])
        
    ssim_mean = np.mean(ssim)
    ssim_std = np.std(ssim)
    
    return ssim_mean, ssim_std

def psnr_vols(hr_vol, sr_vol):
    psnr = [np.zeros(hr_vol[i].shape[2]) for i in range(len(hr_vol))]
    for i in range(len(hr_vol)):
        for j in range(len(psnr[i])):
            psnr[i][j] = skim.peak_signal_noise_ratio(hr_vol[i][:,:,j], sr_vol[i][:,:,j])
    psnr_mean = np.mean(np.concatenate(psnr))
    psnr_std = np.std(np.concatenate(psnr))
    
    return psnr_mean, psnr_std

def ssim_vols(hr_vol, sr_vol):
    ssim = [np.zeros(hr_vol[i].shape[2]) for i in range(len(hr_vol))]
    for i in range(len(hr_vol)):
        for j in range(len(ssim[i])):
            ssim[i][j] = skim.structural_similarity(hr_vol[i][:,:,j], sr_vol[i][:,:,j])
    ssim_mean = np.mean(np.concatenate(ssim))
    ssim_std = np.std(np.concatenate(ssim))
    
    return ssim_mean, ssim_std

## Bone metrics
def otsu_thresh(vol):
    return vol > threshold_otsu(vol)

def bvtv_slice(im_bin, mask):
    tv = np.sum(mask)
    bv = np.sum(im_bin[mask>0])
    
    return bv / tv

def bvtv_vol(vol_bin):
    tv = vol_bin.shape[0]*vol_bin.shape[1]*vol_bin.shape[2]
    bv = np.sum(vol_bin)
    
    return bv / tv

def bvtv_dif_vol(hr_vol_bin, sr_vol_bin):
    bvtv_dif = np.zeros(hr_vol_bin.shape[2])
    for i in range(hr_vol_bin.shape[2]):
        bvtv_dif[i] = bvtv_slice(hr_vol_bin[:,:,i])-bvtv_slice(sr_vol_bin[:,:,i])
    bvtv_mean = np.mean(bvtv_dif)
    bvtv_std = np.std(bvtv_dif)
    
    return bvtv_mean, bvtv_std

def loc_thck(vol_bin):
    return lt.local_thickness(vol_bin, scale=0.5)

def loc_thck_slice(im_bin):
    return lt.local_thickness(im_bin, scale=0.5)

def loc_thck_dif_vol(hr_vol_bin, sr_vol_bin):
    loc_thck_dif = np.zeros(hr_vol_bin.shape[2])
    for i in range(hr_vol_bin.shape[2]):
        loc_thck_dif[i] = loc_thck_slice(hr_vol_bin[:,:,i])-loc_thck_slice(sr_vol_bin[:,:,i])
    loc_thck_mean = np.mean(loc_thck_dif)
    loc_thck_std = np.std(loc_thck_dif)
    
    return loc_thck_mean, loc_thck_std