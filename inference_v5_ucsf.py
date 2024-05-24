## Ver. 5.0: Crop outputted SR images to original non-padded size.

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
    "--bone_no",
    nargs="*",
    type=str,                  
    default=["SP02-01", "SP03-01", "SP04-01", "SP05-01"],
    help="Bones to do inference on"
)
args = CLI.parse_args()

bone_no = args.bone_no
data_type = args.data_type
model_type = args.model_type
pix_loss_type = args.pix_loss_type
cont_loss = args.cont_loss
config = data_type + "_" + model_type + "_" + pix_loss_type + "_" + cont_loss

print(f'Testing on bone no.: {bone_no} \n')
print(f'Training data type: {data_type} \n')
print(f'Model architecture: {model_type} \n')
print(f'Pixel loss type: {pix_loss_type} \n')
print(f'Content loss: {cont_loss} \n')

if model_type == "ESRGAN":
    generator = ESRGenerator(channels=SR_config.IN_C, filters=64, num_res_blocks=SR_config.NUM_RES_UNITS) 
else:
    generator = SRGenerator(in_channels=SR_config.IN_C, out_channels=1, n_residual_blocks=SR_config.NUM_RES_UNITS)

path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/Results/model/" + config + "/model.pt"
generator.load_state_dict(torch.load(path))
generator = generator.cuda()
generator.eval()

dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

for i in range(len(bone_no)):
    
    hr_path = dataset_path + bone_no[i] + "/mct/vol/" + bone_no[i] + "_padded.npy"
    if data_type == "real":
        lr_path = dataset_path + bone_no[i] + "/XCT/vol/" + bone_no[i] + "_padded.npy"
    else:
        lr_path = dataset_path + bone_no[i] + "/synth/vol/" + bone_no[i] + "_padded.npy"
    img_paths = [[hr_path, lr_path]]

    inference_dataloader = DataLoader(SliceData(img_paths), batch_size=1, drop_last=False, shuffle=False, num_workers=4)
        
    sr_vol = np.zeros(SR_config.VOL_SHAPES_UCSF[bone_no[i]],dtype=np.float32)
    sr_vol_bin = np.zeros(SR_config.VOL_SHAPES_UCSF[bone_no[i]],dtype=np.uint8)
    
    org_x = SR_config.VOL_SHAPES_PADDED_UCSF[bone_no[i]][0]
    org_y = SR_config.VOL_SHAPES_PADDED_UCSF[bone_no[i]][1]
    dim_x = SR_config.VOL_SHAPES_UCSF[bone_no[i]][0]
    dim_y = SR_config.VOL_SHAPES_UCSF[bone_no[i]][1]

    print(f'Doing inference...')
    for idx, img in enumerate(inference_dataloader):
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
    # sr_vol = sr_vol.astype(np.float16)
    # sr_vol_bin = sr_vol_bin.astype(np.uint8)
    print(f'Inference completed.')
    
    print(f'Saving SR and binarized SR volumes...')
    save_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/Results/SR/" + bone_no[i] + "/" + config + "/"
    np.save(save_path + bone_no[i] + ".npy",sr_vol)
    np.save(save_path + bone_no[i] + "_binary.npy",sr_vol_bin)
    sr_vol = sr_vol.transpose(2, 0, 1)              #Transpose back to correct axis order for FIJI
    sr_vol_bin = sr_vol_bin.transpose(2, 0, 1)
    tifffile.imwrite(save_path + bone_no[i] + ".tif",sr_vol)
    tifffile.imwrite(save_path + bone_no[i] + "_binary.tif",sr_vol_bin)
    print(f'Saving complete.')