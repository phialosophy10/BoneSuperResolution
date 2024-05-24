## Ver. 5.0: Crop outputted SR images to original non-padded size.

# %% Packages
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
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

# %%
%matplotlib tk

# %% Get command line arguments
# CLI=argparse.ArgumentParser()
# CLI.add_argument(
#     "data_type",
#     type=str,
#     choices=["real", "synth"],
#     default="real",
#     help="input data type (real or synthetic)",
# )
# CLI.add_argument(
#     "model_type",
#     type=str,
#     choices=["ESRGAN", "SRGAN"],
#     default="ESRGAN",
#     help="model architecture (ESRGAN or SRGAN)",
# )
# CLI.add_argument(
#     "pix_loss_type",
#     type=str,
#     choices=["L1", "MSE"],
#     default="L1",
#     help="Type of pixel-loss (L1 or MSE)",
# )
# CLI.add_argument(
#     "cont_loss",
#     type=str,
#     choices=["cont", "no_cont"],
#     default="cont",
#     help="Whether to use content loss or not",
# )
# CLI.add_argument(
#     "--femur_no",
#     nargs="*",
#     type=str,                  
#     default=["002", "086", "138"],
#     help="Bones to do inference on"
# )
# args = CLI.parse_args()

# femur_no = args.femur_no
# data_type = args.data_type
# model_type = args.model_type
# pix_loss_type = args.pix_loss_type
# cont_loss = args.cont_loss
# config = data_type + "_" + model_type + "_" + pix_loss_type + "_" + cont_loss

femur_no = ["002"]
data_type = "real"
model_type = "SRGAN"
pix_loss_type = "L1"
cont_loss = "cont"
config = data_type + "_" + model_type + "_" + pix_loss_type + "_" + cont_loss

print(f'Testing on femur no.: {femur_no} \n')
print(f'Training data type: {data_type} \n')
print(f'Model architecture: {model_type} \n')
print(f'Pixel loss type: {pix_loss_type} \n')
print(f'Content loss: {cont_loss} \n')

if model_type == "ESRGAN":
    generator = ESRGenerator(channels=SR_config.IN_C, filters=64, num_res_blocks=SR_config.NUM_RES_UNITS) 
else:
    generator = SRGenerator(in_channels=SR_config.IN_C, out_channels=1, n_residual_blocks=SR_config.NUM_RES_UNITS)

path = "/work3/soeba/HALOS/Results/model/" + data_type + "_" + model_type + "_" + pix_loss_type + "_" + cont_loss + "/model.pt"
generator.load_state_dict(torch.load(path))
generator = generator.cuda()
generator.eval()

dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

for i in range(len(femur_no)):
    
    hr_path = dataset_path + "femur_" + femur_no[i] + "/micro/volume/f_" + femur_no[i] + "_padded.npy"
    if data_type == "real":
        lr_path = dataset_path + "femur_" + femur_no[i] + "/clinical/volume/f_" + femur_no[i] + "_padded.npy"
    else:
        lr_path = dataset_path + "femur_" + femur_no[i] + "/synth/volume/f_" + femur_no[i] + "_padded.npy"
    img_paths = [[hr_path, lr_path]]

    inference_dataloader = DataLoader(SliceData(img_paths), batch_size=1, drop_last=False, shuffle=False, num_workers=4)
        
    sr_vol = np.zeros(SR_config.VOL_SHAPES[femur_no[i]],dtype=np.float32)
    sr_vol_bin = np.zeros(SR_config.VOL_SHAPES[femur_no[i]],dtype=np.uint8)
    
    org_x = SR_config.VOL_SHAPES_PADDED[femur_no[i]][0]
    org_y = SR_config.VOL_SHAPES_PADDED[femur_no[i]][1]
    dim_x = SR_config.VOL_SHAPES[femur_no[i]][0]
    dim_y = SR_config.VOL_SHAPES[femur_no[i]][1]

    print(f'Doing inference...')
    for idx, img in enumerate(inference_dataloader):
        if idx < 100:
            pass
        elif idx == 100:          
            with torch.inference_mode():
                with torch.autocast("cuda"):
                    img_lr = img["lr"].type(Tensor)
                    img_sr = generator(img_lr)
                    img_sr = img_sr[0][0].cpu().detach().numpy()
                    
                    
                    img_sr = img_sr[(org_x-dim_x)//2:(org_x-dim_x)//2+dim_x,(org_y-dim_y)//2:(org_y-dim_y)//2+dim_y]
                    
                    img_sr = img_sr.astype(np.float32)
                    
                    # Otsu threshold
                    sr_bin = img_sr > threshold_otsu(img_sr[np.isfinite(img_sr)])
                    
                    # Save SR slice in volume
                    v_min = np.min(img_sr)
                    v_max = np.max(img_sr)
                    sr_vol[:,:,idx] = (img_sr-v_min)/(v_max-v_min)
                    sr_vol_bin[:,:,idx] = sr_bin.astype(np.uint8)
        else:
            break
        
    
    # sr_vol = sr_vol.astype(np.uint8)
    # sr_vol_bin = sr_vol_bin.astype(np.uint8)
    # print(f'Inference completed.')
    
    # print(f'Saving SR and binarized SR volumes...')
    # save_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/SR/" + femur_no[i] + "/" + config + "/"
    # np.save(save_path + "f_" + femur_no[i] + ".npy",sr_vol)
    # np.save(save_path + "f_" + femur_no[i] + "_binary.npy",sr_vol_bin)
    # sr_vol = sr_vol.transpose(2, 0, 1)              #Transpose back to correct axis order for FIJI
    # sr_vol_bin = sr_vol_bin.transpose(2, 0, 1)
    # tifffile.imwrite(save_path + "f_" + femur_no[i] + ".tif",sr_vol)
    # tifffile.imwrite(save_path + "f_" + femur_no[i] + "_binary.tif",sr_vol_bin)
    # print(f'Saving complete.')
# %%
with torch.inference_mode():
    with torch.autocast("cuda"):
        img_sr = generator(im_torch.type(Tensor))
# %%
plt.imshow(img_sr[0][0].cpu().detach().numpy(),cmap='gray')
plt.show()
# %%
