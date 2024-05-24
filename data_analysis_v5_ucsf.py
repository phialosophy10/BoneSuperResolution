## Ver. 3.0: Filter out empty slices (acc. to mask), so they don't mess up calculations

# %% Packages
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import argparse
import SR_config
import loss_functions
import evaluation_metrics
import utils
import torch
from models import SRGenerator, ESRGenerator
from torch.utils.data import DataLoader
from datasets import SliceData, PatchData
import skimage.metrics as skim
from skimage.filters import threshold_otsu
import localthickness as lt
import tifffile
import argparse
#%matplotlib tk

# %% Get command line arguments
CLI=argparse.ArgumentParser()
CLI.add_argument(
    "data_type",
    type=str,
    choices=["real", "synth"],
    default="real",
    help="input data type (real or synthetic)",
)
CLI.add_argument(
    "model_type",
    type=str,
    choices=["ESRGAN", "SRGAN"],
    default="ESRGAN",
    help="model architecture (ESRGAN or SRGAN)",
)
CLI.add_argument(
    "pix_loss_type",
    type=str,
    choices=["L1", "MSE"],
    default="L1",
    help="Type of pixel-loss (L1 or MSE)",
)
CLI.add_argument(
    "cont_loss",
    type=str,
    choices=["cont", "no_cont"],
    default="cont",
    help="Whether to use content loss or not",
)
CLI.add_argument(
    "--bone_no",
    nargs="*",
    type=str,                  
    default=["SP02-01", "SP03-01", "SP04-01", "SP05-01"],
    help="Bones to do analysis on"
)
args = CLI.parse_args()

def normalize(im,mask):
    im_mean = im[mask>0].mean()
    im_std = im[mask>0].std()
    im = ((im - im_mean) / im_std)
    # return np.clip(im, 0, 1)
    return im*mask

bone_no = args.bone_no
data_type = args.data_type
model_type = args.model_type
pix_loss_type = args.pix_loss_type
cont_loss = args.cont_loss
# bone_no = ["SP02-01"]
# data_type = "real"
# model_type = "SRGAN"
# pix_loss_type = "L1"
# cont_loss = "cont"
config = data_type + "_" + model_type + "_" + pix_loss_type + "_" + cont_loss

print(f'Testing on bone no.: {bone_no} \n')
print(f'Training data type: {data_type} \n')
print(f'Model architecture: {model_type} \n')
print(f'Pixel loss type: {pix_loss_type} \n')
print(f'Content loss: {cont_loss} \n')

hr_root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/"
sr_root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/Results/SR/"

non_nan_all = []

# %% SSIM, PSNR
ssim_all = []
psnr_all = []

print(f'Calculating PSNR and SSIM...')
for i in range(len(bone_no)):
    hr_vol = np.load(hr_root_path + bone_no[i] + "/mct/vol/" + bone_no[i] + ".npy")
    sr_vol = np.load(sr_root_path + bone_no[i] + "/" + config + "/" + bone_no[i] + ".npy")
    sr_vol = sr_vol / 255
    mask = np.load(hr_root_path + bone_no[i] + "/mask/vol/" + bone_no[i] + "_dilated.npy")
    
    non_nan = []
    for j in range(mask.shape[2]):
        if np.sum(hr_vol[:,:,j]*mask[:,:,j]) != 0 and np.sum(sr_vol[:,:,j]*mask[:,:,j]) != 0:
            non_nan.append(j)
    print(f'bone {bone_no[i]} had {mask.shape[2]-len(non_nan)} empty masked slices that were not included in calculations.')
    non_nan_all.append(non_nan)
    
    ssim = np.zeros(len(non_nan))
    psnr = np.zeros(len(non_nan))
    
    count = 0
    for idx in non_nan:
        # SSIM, PSNR
        im_hr_norm = normalize(hr_vol[:,:,idx],mask[:,:,idx])
        im_sr_norm = normalize(sr_vol[:,:,idx],mask[:,:,idx])
        data_range=im_hr_norm.max()-im_hr_norm.min()
        _, S = skim.structural_similarity(im_hr_norm, im_sr_norm, full=True, data_range=data_range)
        ssim[count] = S[mask[:,:,idx]>0].mean()
        psnr[count] = skim.peak_signal_noise_ratio(im_hr_norm[mask[:,:,idx]>0], im_sr_norm[mask[:,:,idx]>0], data_range=data_range)
        count += 1
        
    ssim_all.append(ssim)
    psnr_all.append(psnr)
    
print(f'PSNR and SSIM calculations done.')

# %% BV/TV, Thickness
bvtv_hr_trab_all = []
bvtv_sr_trab_all = []
median_hr_trab_all = []
median_sr_trab_all = []
bvtv_hr_cort_all = []
bvtv_sr_cort_all = []
median_hr_cort_all = []
median_sr_cort_all = []

print(f'Calculating BV/TV and thickness...')
for i in range(len(bone_no)):
    hr_vol = np.load(hr_root_path + bone_no[i] + "/mct/vol/" + bone_no[i] + "_binary.npy")
    sr_vol = np.load(sr_root_path + bone_no[i] + "/" + config + "/" + bone_no[i] + "_binary.npy")
    mask_trab = np.load(hr_root_path + bone_no[i] + "/mask/vol/" + bone_no[i] + "_TRAB.npy")
    mask_cort = np.load(hr_root_path + bone_no[i] + "/mask/vol/" + bone_no[i] + "_CORT.npy")
    # mask = mask_trab + mask_cort
    
    # print(f'hr_vol shape: {hr_vol.shape}')
    # print(f'mask shape: {mask.shape}')
    
    non_nan = non_nan_all[i]
    # non_nan = []
    # for j in range(mask.shape[2]):
    #     if np.sum(hr_vol[:,:,j]*mask[:,:,j]) != 0 and np.sum(sr_vol[:,:,j]*mask[:,:,j]) != 0:
    #         non_nan.append(j)
    print(f'bone {bone_no[i]} had {mask.shape[2]-len(non_nan)} empty masked slices that were not included in calculations.')
    
    bvtv_hr_trab = np.zeros(len(non_nan))
    bvtv_sr_trab = np.zeros(len(non_nan))
    median_hr_trab = np.zeros(len(non_nan))
    median_sr_trab = np.zeros(len(non_nan))
    bvtv_hr_cort = np.zeros(len(non_nan))
    bvtv_sr_cort = np.zeros(len(non_nan))
    median_hr_cort = np.zeros(len(non_nan))
    median_sr_cort = np.zeros(len(non_nan))
    
    # mask = (mask / 255).astype(np.uint8)
    
    count = 0
    for idx in non_nan:
        # BVTV
        bvtv_hr_trab[count] = evaluation_metrics.bvtv_slice(hr_vol[:,:,idx], mask_trab[:,:,idx])
        bvtv_sr_trab[count] = evaluation_metrics.bvtv_slice(sr_vol[:,:,idx], mask_trab[:,:,idx])
        bvtv_hr_cort[count] = evaluation_metrics.bvtv_slice(hr_vol[:,:,idx], mask_cort[:,:,idx])
        bvtv_sr_cort[count] = evaluation_metrics.bvtv_slice(sr_vol[:,:,idx], mask_cort[:,:,idx])
        
        # Local thickness
        hr_masked_trab = hr_vol[:,:,idx]*mask_trab[:,:,idx]
        sr_masked_trab = sr_vol[:,:,idx]*mask_trab[:,:,idx]
        thick_hr_trab = lt.local_thickness(hr_masked_trab, scale=0.5)
        thick_sr_trab = lt.local_thickness(sr_masked_trab, scale=0.5)
        thick_hr_trab = thick_hr_trab[thick_hr_trab>0]
        thick_sr_trab = thick_sr_trab[thick_sr_trab>0]
        logthick_hr_trab = np.log(thick_hr_trab)
        logthick_sr_trab = np.log(thick_sr_trab)
        median_hr_trab[count] = np.exp(logthick_hr_trab.mean())
        median_sr_trab[count] = np.exp(logthick_sr_trab.mean())
        
        hr_masked_cort = hr_vol[:,:,idx]*mask_cort[:,:,idx]
        sr_masked_cort = sr_vol[:,:,idx]*mask_cort[:,:,idx]
        thick_hr_cort = lt.local_thickness(hr_masked_cort, scale=0.5)
        thick_sr_cort = lt.local_thickness(sr_masked_cort, scale=0.5)
        thick_hr_cort = thick_hr_cort[thick_hr_cort>0]
        thick_sr_cort = thick_sr_cort[thick_sr_cort>0]
        logthick_hr_cort = np.log(thick_hr_cort)
        logthick_sr_cort = np.log(thick_sr_cort)
        median_hr_cort[count] = np.exp(logthick_hr_cort.mean())
        median_sr_cort[count] = np.exp(logthick_sr_cort.mean())
        
        count += 1
    
    bvtv_hr_trab_all.append(bvtv_hr_trab)
    bvtv_sr_trab_all.append(bvtv_sr_trab)
    median_hr_trab_all.append(median_hr_trab)
    median_sr_trab_all.append(median_sr_trab)
    bvtv_hr_cort_all.append(bvtv_hr_cort)
    bvtv_sr_cort_all.append(bvtv_sr_cort)
    median_hr_cort_all.append(median_hr_cort)
    median_sr_cort_all.append(median_sr_cort)

print(f'BV/TV and thickness calculations completed.')

# for i in range(len(bvtv_hr_all)):
#     print(f'HR BV/TV array for bone {bone_no[i]} has shape {bvtv_hr_all[i].shape}')
#     print(f'SR BV/TV array for bone {bone_no[i]} has shape {bvtv_sr_all[i].shape}')
#     print(f'HR thickness array for bone {bone_no[i]} has shape {median_hr_all[i].shape}')
#     print(f'SR thickness array for bone {bone_no[i]} has shape {median_sr_all[i].shape}')

# %%
print(f'Performing comparative analysis...')
ssim_mean = np.mean(np.concatenate(ssim_all))
ssim_std = np.std(np.concatenate(ssim_all))
psnr_mean = np.mean(np.concatenate(psnr_all))
psnr_std = np.std(np.concatenate(psnr_all))
bvtv_trab_mean = np.mean(np.absolute(np.concatenate(bvtv_hr_trab_all) - np.concatenate(bvtv_sr_trab_all)))
bvtv_trab_std = np.std(np.absolute(np.concatenate(bvtv_hr_trab_all) - np.concatenate(bvtv_sr_trab_all)))
median_trab_mean = np.mean(np.absolute(np.concatenate(median_hr_trab_all) - np.concatenate(median_sr_trab_all)))
median_trab_std = np.std(np.absolute(np.concatenate(median_hr_trab_all) - np.concatenate(median_sr_trab_all)))
bvtv_cort_mean = np.mean(np.absolute(np.concatenate(bvtv_hr_cort_all) - np.concatenate(bvtv_sr_cort_all)))
bvtv_cort_std = np.std(np.absolute(np.concatenate(bvtv_hr_cort_all) - np.concatenate(bvtv_sr_cort_all)))
median_cort_mean = np.mean(np.absolute(np.concatenate(median_hr_cort_all) - np.concatenate(median_sr_cort_all)))
median_cort_std = np.std(np.absolute(np.concatenate(median_hr_cort_all) - np.concatenate(median_sr_cort_all)))
print(f'Analysis completed.')

# %% Print statistics
print(f'PSNR between HR and SR slices: {psnr_mean} plus-minus {psnr_std} \n')
print(f'SSIM between HR and SR slices: {ssim_mean} plus-minus {ssim_std} \n')
print(f'Difference in BV/TV between HR and SR slices in trabecular compartment: {bvtv_trab_mean} plus-minus {bvtv_trab_std} \n')
print(f'Difference in median local thickness between HR and SR slices in trabecular compartment: {median_trab_mean} plus-minus {median_trab_std} \n')
print(f'Difference in BV/TV between HR and SR slices in cortical compartment: {bvtv_cort_mean} plus-minus {bvtv_cort_std} \n')
print(f'Difference in median local thickness between HR and SR slices in cortical compartment: {median_cort_mean} plus-minus {median_cort_std} \n')
# %% Plot
print(f'Making figure...')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(1, 2)
for i in range(len(bone_no)):
    axs[0].scatter(bvtv_hr_trab_all[i], bvtv_sr_trab_all[i], color=colors[i], marker = 'o', s=1, label=bone_no[i])
    axs[0].plot([0, 0.4],[0, 0.4], 'k-')
    axs[0].legend(loc="lower right")
    axs[0].set_title('BV/TV - trab. comp.')
    axs[1].scatter(median_hr_trab_all[i], median_sr_trab_all[i], color=colors[i], marker = 'o', s=1, label=bone_no[i])
    axs[1].plot([0, 2000],[0, 2000], 'k-')
    axs[1].legend(loc="lower right")
    axs[1].set_title('Median thickness - trab. comp.')
#plt.show()
plt.savefig("/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/Results/model/" + config + "/analysis_figure_trab.png")
fig, axs = plt.subplots(1, 2)
for i in range(len(bone_no)):
    axs[0].scatter(bvtv_hr_cort_all[i], bvtv_sr_cort_all[i], color=colors[i], marker = 'o', s=1, label=bone_no[i])
    axs[0].plot([0, 0.4],[0, 0.4], 'k-')
    axs[0].legend(loc="lower right")
    axs[0].set_title('BV/TV - cort. comp.')
    axs[1].scatter(median_hr_cort_all[i], median_sr_cort_all[i], color=colors[i], marker = 'o', s=1, label=bone_no[i])
    axs[1].plot([0, 400],[0, 400], 'k-')
    axs[1].legend(loc="lower right")
    axs[1].set_title('Median thickness - cort. comp.')
#plt.show()
plt.savefig("/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/Results/model/" + config + "/analysis_figure_cort.png")
print(f'Saved BV/TV-Thickness figures at </dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/Results/model/{config}/analysis_figure_[comp].png>')
# %%
