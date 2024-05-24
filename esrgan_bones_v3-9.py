## Ver. 3.9: New root path (new data structure)

# %% Packages
import random
import glob
import wandb
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import utils
import SR_config
from datasets import SliceData_v2, PatchData_v3
from models import FeatureExtractor, Discriminator2D, SRGenerator, ESRGenerator
from loss_functions import VGGLoss
from evaluation_metrics import psnr_vols, ssim_vols, otsu_thresh

# %% Settings
random.seed(42)

use_amp = True
save_models = True

args = utils.get_cmd_args()
femur_no = args.train
femur_no_test = args.test
if(args.loss_coeffs[2] == 0):
    cont_loss_str = "no_cont"
else:
    cont_loss_str = "cont"
config = args.data_type + "_" + args.model_type + "_" + args.pix_loss + "_" + cont_loss_str

channels = 1
hr_height = SR_config.PATCH_SIZE                     # 64, 128, 256
hr_width = SR_config.PATCH_SIZE                      # 64, 128, 256
hr_shape = (hr_height, hr_width)

opt_adam_b1 = 0.9 # adam: decay of first order momentum of gradient
opt_adam_b2 = 0.999 # adam: decay of second order momentum of gradient

dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/SR_proj/data/DTU/"

go_cuda = torch.cuda.is_available()

TOTAL_GPU_MEM = torch.cuda.get_device_properties(0).total_memory/10**9

# Logging with 'Weights and Biases (WandB)'
log_wandb = True
if log_wandb:
    wandb.init(project="esrgan_bones_3.9", entity="phialosophy", config={})
    DT = datetime.now()
    wandb.run.name = "train" + str(len(femur_no)) + "_" + str(args.n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")
    wandb.config.update({
        "model type": args.model_type,
        "data type": args.data_type,
        "learning_rate": args.lr,
        "epochs": args.n_epochs,
        "batch_size": args.batch_size, #args.batch_size, #SR_config.BATCH_SIZE
        "femur_train": femur_no,
        "femur_test": femur_no_test,
        "pixel-loss type": args.pix_loss,
        "no. of res. blocks": SR_config.NUM_RES_UNITS,
        "loss_coefficients (adv, pix, cont)": (args.loss_coeffs[0], args.loss_coeffs[1], args.loss_coeffs[2]),
    })

# %% Make dataset paths
hr_paths_train, lr_paths_train, mask_paths_train = [], [], []
for i in range(len(femur_no)):
    hr_paths_train.append(dataset_path + "f_" + femur_no[i] + "/HR/f_" + femur_no[i] + ".npy")
    if args.data_type == "real":
        lr_paths_train.append(dataset_path + "f_" + femur_no[i] + "/LR/f_" + femur_no[i] + ".npy")
    else:
        lr_paths_train.append(dataset_path + "f_" + femur_no[i] + "/SY/f_" + femur_no[i] + ".npy")
    mask_paths_train.append(dataset_path + "f_" + femur_no[i] + "/MS/f_" + femur_no[i] + ".npy")
img_paths_train = []
for i in range(len(hr_paths_train)):
    img_paths_train.append([hr_paths_train[i], lr_paths_train[i], mask_paths_train[i]])

hr_paths_test, lr_paths_test = [], []
for i in range(len(femur_no_test)):
    hr_paths_test.append(dataset_path + "f_" + femur_no_test[i] + "/HR/f_" + femur_no_test[i] + ".npy")
    if args.data_type == "real":
        lr_paths_test.append(dataset_path + "f_" + femur_no_test[i] + "/LR/f_" + femur_no_test[i] + ".npy")
    else:
        lr_paths_test.append(dataset_path + "f_" + femur_no_test[i] + "/SY/f_" + femur_no_test[i] + ".npy")
img_paths_test = []
for i in range(len(hr_paths_test)):
    img_paths_test.append([hr_paths_test[i], lr_paths_test[i]])

# %% Datasets and dataloaders
train_dataloader = DataLoader(PatchData_v3(file_paths=img_paths_train, patch_size=SR_config.PATCH_SIZE, num_patches=20000), batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4)
test_dataloader = DataLoader(SliceData_v2(file_paths=img_paths_test, pad=True, num_patches=6), batch_size=SR_config.BATCH_SIZE_TEST, drop_last=True, shuffle=False, num_workers=4)

# %% Train model 
# Initialize generator and discriminator
if args.model_type == "ESRGAN":
    generator = ESRGenerator(channels=SR_config.IN_C, filters=64, num_res_blocks=SR_config.NUM_RES_UNITS) 
elif args.model_type == "SRGAN":
    generator = SRGenerator(in_channels=SR_config.IN_C, out_channels=1, n_residual_blocks=SR_config.NUM_RES_UNITS)
else:
    generator = ESRGenerator(channels=SR_config.IN_C, filters=64, num_res_blocks=SR_config.NUM_RES_UNITS)
discriminator = Discriminator2D(input_shape=(SR_config.IN_C, *hr_shape))

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = VGGLoss(layer_idx=35)
if args.pix_loss == "L1":
    criterion_pixel = torch.nn.L1Loss()
elif args.pix_loss == "MSE":
    criterion_pixel = torch.nn.MSELoss()
else:
    criterion_pixel = torch.nn.L1Loss()

# Send to GPU
if go_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    criterion_pixel = criterion_pixel.cuda()

Tensor = torch.cuda.FloatTensor if go_cuda else torch.Tensor

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(opt_adam_b1, opt_adam_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(opt_adam_b1, opt_adam_b2))

timing = []
scaler = torch.cuda.amp.GradScaler()

# Train and test loop
utils.start_timer()
for epoch in range(args.n_epochs):
    start_time = time()
    
    ### Training
    print(f'Training Epoch {epoch+1}')
    
    torch.cuda.empty_cache()

    gen_loss, disc_loss, gen_count, disc_count = 0, 0, 0, 0
    for batch_idx, imgs in enumerate(train_dataloader):

        # Configure model input
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)
        
        #if (batch_idx+1) % 10 == 0:
        #    print("Memory allocation after batch load: %0.3f Gb / %0.3f Gb" % (torch.cuda.memory_allocated()/10**9, TOTAL_GPU_MEM))
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Generate SR image
            gen_hr = generator(imgs_lr)

            # Pixel loss
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            # Content loss
            loss_content = criterion_content(imgs_hr, gen_hr)
            
            # Adversarial loss
            pred_fake = discriminator(gen_hr)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

            # Total generator loss
            loss_G = args.loss_coeffs[1] * loss_pixel + args.loss_coeffs[2] * loss_content + args.loss_coeffs[0] * loss_GAN
        
        # Generator update
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()
        optimizer_G.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Adversarial loss
            gen_hr = generator(imgs_lr)
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real) - 0.1 * torch.ones_like(pred_real))  
        
            # Total discriminator loss
            loss_D = loss_real + loss_fake
        
        # Discriminator update
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()
        optimizer_D.zero_grad(set_to_none=True)

        # Save losses
        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
        gen_count += 1
        disc_count += 1
        
        if (batch_idx+1) % 100 == 0:
            print(f'Epoch {epoch+1}, batch no. {batch_idx+1}, gen. loss: {loss_G.item()}, disc. loss: {loss_D.item()}')
    if save_models:
        torch.save(generator.state_dict(), "/work3/soeba/HALOS/Results/model/" + config + "/model.pt")
    print(f'Saved model at </work3/soeba/HALOS/Results/model/{config}/model.pt>')
    
    #max_memory_reserved = torch.cuda.max_memory_reserved()
    #print("Maximum memory reserved during epoch: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, TOTAL_GPU_MEM))

    # Print loss after each epoch
    print(f'Epoch {epoch+1}, batch no. {batch_idx+1}, gen. loss: {gen_loss/gen_count}, disc. loss: {disc_loss/disc_count}')
    if log_wandb:
        wandb.log({"loss/gen_loss": gen_loss/gen_count, "loss/disc_loss": disc_loss/disc_count})
    
    ### Testing
    print(f'Testing Epoch {epoch+1}')
    for batch_idx, imgs in enumerate(test_dataloader):
        with torch.inference_mode():
            # Create grid of images with LR, HR and SR (for last batch of each epoch)
            # if batch_idx+1 == len(test_dataloader)-1: 
            generator.eval()

            # Configure model input
            imgs_lr = imgs["lr"].type(Tensor)
            imgs_hr = imgs["hr"].type(Tensor)

            # Generate SR images
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False): #use_amp
                gen_hr = generator(imgs_lr)

            if log_wandb:
                # img_grid1, img_grid2 = utils.make_dif_thck_ims_v3(imgs_hr[-5:], imgs_lr[-5:], gen_hr[-5:])
                img_grid1, img_grid2 = utils.make_dif_thck_ims_v3(imgs_hr[-5:], imgs_lr[-5:], gen_hr[-5:])
                image1 = wandb.Image(img_grid1, caption="HR - LR - SR - dif")
                wandb.log({"images": image1})
                image2 = wandb.Image(img_grid2, caption="thick HR - hist HR - thick SR - hist SR")
                wandb.log({"thickness and histograms": image2})
    timing.append(time() - start_time)
utils.end_timer_and_print("AMP:")

print(f'Finished training!')
for n in range(args.n_epochs):
    print(f'Epoch no. {n+1} took {timing[n]/60} minutes')

# Write job summary to log file
log_info = [
    "Log info for latest run.",
    f'Timestamp: {DT.strftime("%d/%m_%H:%M")} \n',
    f'Number of epochs: {args.n_epochs} \n',
    f'Batch size for training: {args.batch_size} \n', #{args.batch_size} #{SR_config.BATCH_SIZE}
    f'Learning rate: {args.lr} \n',
    f'Coefficient for adv. loss term: {args.loss_coeffs[0]} \n',
    f'Coefficient for pix. loss term: {args.loss_coeffs[1]} \n',
    f'Coefficient for cont. loss term: {args.loss_coeffs[2]} \n',
    f'Model architecture (ESRGAN/SRGAN): {args.model_type} \n',
    f'Type of training data (synth/real): {args.data_type} \n',
    f'Number of residual blocks: {SR_config.NUM_RES_UNITS} \n',
    f'Pixel loss-type (MSE/L1): {args.pix_loss} \n'
]

with open("/work3/soeba/HALOS/Results/model/" + config + "/latest_run.txt", "w") as f:
    f.writelines(log_info)
print("Saved log file!")
