## Ver. 3.0: Added check for empty masks

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
    "--femur_no",
    nargs="*",
    type=str,                  
    default=["002", "086", "138"],
    help="Bones to do analysis on"
)
args = CLI.parse_args()

def normalize(im,mask):
    im_mean = im[mask>0].mean()
    im_std = im[mask>0].std()
    im = ((im - im_mean) / im_std)
    # return np.clip(im, 0, 1)
    return im*mask

femur_no = args.femur_no
data_type = args.data_type
model_type = args.model_type
pix_loss_type = args.pix_loss_type
cont_loss = args.cont_loss
config = data_type + "_" + model_type + "_" + pix_loss_type + "_" + cont_loss

print(f'Testing on femur no.: {femur_no} \n')
print(f'Training data type: {data_type} \n')
print(f'Model architecture: {model_type} \n')
print(f'Pixel loss type: {pix_loss_type} \n')
print(f'Content loss: {cont_loss} \n')

hr_root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"
sr_root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/SR/"

ssim = [np.zeros(SR_config.VOL_SHAPES[femur][2]) for femur in femur_no]
psnr = [np.zeros(SR_config.VOL_SHAPES[femur][2]) for femur in femur_no]

print(f'Calculating PSNR and SSIM...')
for i in range(len(femur_no)):
    hr_vol = np.load(hr_root_path + "femur_" + femur_no[i] + "/micro/volume/f_" + femur_no[i] + ".npy")
    sr_vol = np.load(sr_root_path + femur_no[i] + "/" + config + "/f_" + femur_no[i] + ".npy")
    mask = np.load(hr_root_path + "femur_" + femur_no[i] + "/mask/volume/f_" + femur_no[i] + "_dilated.npy")
    
    for idx in range(sr_vol.shape[2]):
        # SSIM, PSNR
        im_hr_norm = normalize(hr_vol[:,:,idx],mask[:,:,idx])
        im_sr_norm = normalize(sr_vol[:,:,idx],mask[:,:,idx])
        data_range=im_hr_norm.max()-im_hr_norm.min()
        _, S = skim.structural_similarity(im_hr_norm, im_sr_norm, full=True, data_range=data_range)
        ssim[i][idx] = S[mask[:,:,idx]>0].mean()
        psnr[i][idx] = skim.peak_signal_noise_ratio(im_hr_norm[mask[:,:,idx]>0], im_sr_norm[mask[:,:,idx]>0], data_range=data_range)
print(f'PSNR and SSIM calculations done.')

bvtv_hr_all = []
bvtv_sr_all = []
median_hr_all = []
median_sr_all = []

print(f'Calculating BV/TV and thickness...')
for i in range(len(femur_no)):
    hr_vol = np.load(hr_root_path + "femur_" + femur_no[i] + "/micro/volume/f_" + femur_no[i] + "_binary.npy")
    sr_vol = np.load(sr_root_path + femur_no[i] + "/" + config + "/f_" + femur_no[i] + "_binary.npy")
    if data_type == "synth":
        sr_vol = sr_vol.astype(np.uint8)
    mask = np.load(hr_root_path + "femur_" + femur_no[i] + "/mask/volume/f_" + femur_no[i] + "_dilated.npy")
    
    non_nan = []
    for j in range(mask.shape[2]):
        if np.sum(hr_vol[:,:,j]*mask[:,:,j]) != 0 and np.sum(sr_vol[:,:,j]*mask[:,:,j]) != 0:
            non_nan.append(j)
    print(f'femur {femur_no[i]} had {mask.shape[2]-len(non_nan)} epmty masked slices that were not included in calculations.')
    
    bvtv_hr = np.zeros(len(non_nan))
    bvtv_sr = np.zeros(len(non_nan))
    median_hr = np.zeros(len(non_nan))
    median_sr = np.zeros(len(non_nan))
    
    count = 0
    for idx in non_nan:
        # BVTV
        bvtv_hr[count] = evaluation_metrics.bvtv_slice(hr_vol[:,:,idx], mask[:,:,idx])
        bvtv_sr[count] = evaluation_metrics.bvtv_slice(sr_vol[:,:,idx], mask[:,:,idx])
        
        # Local thickness
        hr_masked = hr_vol[:,:,idx]*mask[:,:,idx]
        sr_masked = sr_vol[:,:,idx]*mask[:,:,idx]
        thick_hr = lt.local_thickness(hr_masked, scale=0.5)
        thick_sr = lt.local_thickness(sr_masked, scale=0.5)
        thick_hr = thick_hr[thick_hr>0]
        thick_sr = thick_sr[thick_sr>0]
        logthick_hr = np.log(thick_hr)
        logthick_sr = np.log(thick_sr)
        median_hr[count] = np.exp(logthick_hr.mean())
        median_sr[count] = np.exp(logthick_sr.mean())

        count += 1
    
    bvtv_hr_all.append(bvtv_hr)
    bvtv_sr_all.append(bvtv_sr)
    median_hr_all.append(median_hr)
    median_sr_all.append(median_sr)
    
    print(f'BV/TV and thickness calculated for bone {femur_no[i]}')
print(f'BV/TV and thickness calculations completed.')

print(f'Performing comparative analysis...')
ssim_mean = np.mean(np.concatenate(ssim))
ssim_std = np.std(np.concatenate(ssim))
psnr_mean = np.mean(np.concatenate(psnr))
psnr_std = np.std(np.concatenate(psnr))
bvtv_mean = np.mean(np.absolute(np.concatenate(bvtv_hr_all) - np.concatenate(bvtv_sr_all)))
bvtv_std = np.std(np.absolute(np.concatenate(bvtv_hr_all) - np.concatenate(bvtv_sr_all)))
median_mean = np.mean(np.absolute(np.concatenate(median_hr_all) - np.concatenate(median_sr_all)))
median_std = np.std(np.absolute(np.concatenate(median_hr_all) - np.concatenate(median_sr_all)))
print(f'Analysis completed.')

print(f'Making figure...')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(len(femur_no)):
    ax1.scatter(bvtv_hr_all[i], bvtv_sr_all[i], color=colors[i], marker = '.', s=1, label=femur_no[i])
    ax1.legend(loc="lower right")
    ax1.set_title('BV/TV')
    ax2.scatter(median_hr_all[i], median_sr_all[i], color=colors[i], marker = '.', s=1, label=femur_no[i])
    ax2.legend(loc="lower right")
    ax2.set_title('Median thickness')
plt.savefig("/work3/soeba/HALOS/Results/model/" + config + "/analysis_figure.png")
print(f'Saved BV/TV-Thickness figure at </work3/soeba/HALOS/Results/model/{config}/analysis_figure.png>')

print(f'PSNR between HR and SR slices: {psnr_mean} plus-minus {psnr_std} \n')
print(f'SSIM between HR and SR slices: {ssim_mean} plus-minus {ssim_std} \n')
print(f'Difference in BV/TV between HR and SR slices: {bvtv_mean} plus-minus {bvtv_std} \n')
print(f'Difference in median local thickness between HR and SR slices: {median_mean} plus-minus {median_std} \n')