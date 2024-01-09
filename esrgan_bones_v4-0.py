# %% Packages
import numpy as np
import pandas as pd
import os, math, sys
import random
import glob
import utils
import models
import SR_config
import loss_functions
import cv2
import wandb

import functools
import torch
import torch.nn as nn
import torchio as tio
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from datasets import BlockData

import matplotlib.pyplot as plt

import porespy as ps
ps.visualization.set_mpl_style()

from PIL import Image

from datetime import datetime

random.seed(42)

# %% Settings
log_wandb = True

if log_wandb:
    wandb.init(project="esrgan_bones_4.0", entity="phialosophy", config={})

# Variable settings
femur_no = ["001"]              # [001,002,013,015,021,026,074,075,083,086,138,164,172]      # choose which bone(s) to train on
femur_no_test = "001"                                           # choose which bone to test on

n_epochs = 5
batch_size = 4
lr = 0.0001 #0.0001
block_shape = (SR_config.PATCH_SIZE, SR_config.PATCH_SIZE, SR_config.PATCH_SIZE)

dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"

if log_wandb:
    DT = datetime.now()
    wandb.run.name = "ESRGAN_" + str(n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")
    wandb.config.update({
        "learning_rate": lr,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "femur_train": femur_no,
        "femur_test": femur_no_test,
        "loss_coefficients (MSE, L1, VGG3D, ADV)": (SR_config.LOSS_WEIGHTS["MSE"], SR_config.LOSS_WEIGHTS["L1"], SR_config.LOSS_WEIGHTS["VGG3D"], SR_config.LOSS_WEIGHTS["ADV"]),
    })

# adam: decay of first order momentum of gradient
b1 = 0.9
# adam: decay of second order momentum of gradient
b2 = 0.999

# %% Make dictionaries with dataset paths

hr_paths_train, lr_paths_train = [], []
for i in range(len(femur_no)):
    hr_paths_train += sorted(glob.glob(dataset_path + "femur_" + femur_no[i] + "/micro/blocks/f_" + femur_no[i] + "*.*"))
    lr_paths_train += sorted(glob.glob(dataset_path + "femur_" + femur_no[i] + "/clinical/blocks/f_" + femur_no[i] + "*.*"))

hr_paths_test = sorted(glob.glob(dataset_path + "femur_" + femur_no_test + "/micro/blocks/f_" + femur_no_test + "*.*"))
lr_paths_test = sorted(glob.glob(dataset_path + "femur_" + femur_no_test + "/clinical/blocks/f_" + femur_no_test + "*.*"))

img_paths_train = []
img_paths_test = []
for i in range(len(hr_paths_train)):
    img_paths_train.append([hr_paths_train[i], lr_paths_train[i]])
for i in range(len(hr_paths_test)):
    img_paths_test.append([hr_paths_test[i], lr_paths_test[i]])

# %% Train and test loaders
train_dataloader = DataLoader(BlockData(img_paths_train, shape=block_shape), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=1)
test_dataloader = DataLoader(BlockData(img_paths_test, shape=block_shape), batch_size=batch_size, drop_last=False, shuffle=True, num_workers=1)

# %% Train model 

# Initialize generator and discriminator
n_conv_vec = [64, 64, 128, 128, 256, 256, 512, 512]
n_dense = [512, 1]  # n_dense = [1024, 1]

generator = models.MultiLevelDenseNet(SR_config.IN_C, SR_config.K_FACTOR, SR_config.K_SIZE).to(SR_config.DEVICE)
discriminator = models.Discriminator3D(SR_config.PATCH_SIZE, SR_config.IN_C, n_conv_vec, n_dense, SR_config.K_SIZE).to(SR_config.DEVICE)
feature_extractor = models.FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
loss_fn_mse = nn.MSELoss()
loss_fn_l1 = nn.L1Loss()
loss_fn_logistic_bce = nn.BCEWithLogitsLoss()
loss_fn_bce = nn.BCELoss()
loss_fn_vgg3D = loss_functions.VGGLoss3D(layer_idx=35)  # Use feature map before activation instead of after

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Tensor = torch.cuda.FloatTensor if SR_config.DEVICE == 'cuda' else torch.Tensor

gen_scaler = torch.cuda.amp.GradScaler()
dis_scaler = torch.cuda.amp.GradScaler()

loss_fn_dict = {
        "MSE_Loss": loss_fn_mse,
        "L1_Loss": loss_fn_l1,
        "BCE_Logistic_Loss": loss_fn_logistic_bce,
        "BCE_Loss": loss_fn_bce,
        "VGG3D_Loss": loss_fn_vgg3D,
        }

def compute_discriminator_loss(prop_real, prop_fake, loss_fn_dict):

    # Formulate Discriminator loss: Max log(D(I_HR)) + log(1 - D(G(I_LR)))
    dis_loss_fake = loss_fn_dict["BCE_Logistic_Loss"](prop_fake, torch.zeros_like(prop_fake))
    dis_loss_real = loss_fn_dict["BCE_Logistic_Loss"](prop_real, torch.ones_like(prop_real) - 0.1 * torch.ones_like(prop_real))  # Added one-side label smoothing
    dis_loss = dis_loss_real + dis_loss_fake

    return dis_loss

def compute_generator_loss(prop_fake, real_hi_res, fake_hi_res, loss_fn_dict, epoch, loss_dict, n_samples):

    # Formulate Generator loss: Min log(I_HR - G(I_LR)) <-> Max log(D(G(I_LR)))
    adv_loss = SR_config.LOSS_WEIGHTS["ADV"] * loss_fn_dict["BCE_Logistic_Loss"](prop_fake, torch.ones_like(prop_fake))

    gen_loss = torch.tensor(0.0).to(SR_config.DEVICE)
    if SR_config.LOSS_WEIGHTS["L1"] != 0:
        gen_loss_L1 = SR_config.LOSS_WEIGHTS["L1"] * loss_fn_dict["L1_Loss"](real_hi_res, fake_hi_res)
        gen_loss += gen_loss_L1
        loss_dict["L1"][epoch] += gen_loss_L1.item() / n_samples
    if SR_config.LOSS_WEIGHTS["MSE"] != 0:
        gen_loss_MSE = SR_config.LOSS_WEIGHTS["MSE"] * loss_fn_dict["MSE_Loss"](real_hi_res, fake_hi_res)
        gen_loss += gen_loss_MSE
        loss_dict["MSE"][epoch] += gen_loss_MSE.item() / n_samples
    if SR_config.LOSS_WEIGHTS["GRAD"] != 0:
        # We should implement this as three 1-D convolutions with gaussian derivatives.
        gen_loss_GRAD = SR_config.LOSS_WEIGHTS["GRAD"] * loss_fn_dict["GRAD_Loss"](real_hi_res, fake_hi_res)
        gen_loss += gen_loss_GRAD
        loss_dict["GRAD"][epoch] += gen_loss_GRAD.item() / n_samples
    if SR_config.LOSS_WEIGHTS["LAPLACE"] != 0:
        gen_loss_LAPLACE = SR_config.LOSS_WEIGHTS["LAPLACE"] * loss_fn_dict["LAPLACE_Loss"](real_hi_res, fake_hi_res)
        gen_loss += gen_loss_LAPLACE[0]
        loss_dict["LAPLACE"][epoch] += gen_loss_LAPLACE[0].item() / n_samples
    if SR_config.LOSS_WEIGHTS["VGG3D"] != 0:
        gen_loss_VGG3D = SR_config.LOSS_WEIGHTS["VGG3D"] * loss_fn_dict["VGG3D_Loss"](real_hi_res, fake_hi_res)
        gen_loss += gen_loss_VGG3D[0]
        loss_dict["VGG3D"][epoch] += gen_loss_VGG3D[0].item() / n_samples
    if SR_config.LOSS_WEIGHTS["TV3D"] != 0:
        gen_loss_TV3D = SR_config.LOSS_WEIGHTS["TV3D"] * loss_fn_dict["TV3D_Loss"](fake_hi_res)
        gen_loss += gen_loss_TV3D
        loss_dict["TV3D"][epoch] += gen_loss_TV3D.item() / n_samples
    if SR_config.LOSS_WEIGHTS["STRUCTURE_TENSOR"] != 0:
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=False):
            gen_loss_STRUCTURE_TENSOR = SR_config.LOSS_WEIGHTS["STRUCTURE_TENSOR"] * loss_fn_dict["STRUCTURE_TENSOR_Loss"](real_hi_res, fake_hi_res)
        gen_loss += gen_loss_STRUCTURE_TENSOR[0]
        loss_dict["STRUCTURE_TENSOR"][epoch] += gen_loss_STRUCTURE_TENSOR[0].item() / n_samples

    gen_loss += adv_loss
    return gen_loss

gen_train_loss_dict = {"L1": torch.zeros(n_epochs), "MSE": torch.zeros(n_epochs), "GRAD": torch.zeros(n_epochs), "LAPLACE": torch.zeros(n_epochs), "VGG3D": torch.zeros(n_epochs), "TV3D": torch.zeros(n_epochs), "STRUCTURE_TENSOR": torch.zeros(n_epochs)}
gen_valid_loss_dict = {"L1": torch.zeros(n_epochs), "MSE": torch.zeros(n_epochs), "GRAD": torch.zeros(n_epochs), "LAPLACE": torch.zeros(n_epochs), "VGG3D": torch.zeros(n_epochs), "TV3D": torch.zeros(n_epochs), "STRUCTURE_TENSOR": torch.zeros(n_epochs)}

sigmoid_func = nn.Sigmoid()

for epoch in range(n_epochs):
    print(f'Training Epoch {epoch+1}')
    
    gen_train_loss, dis_train_loss = 0, 0
    loss_count_train_tmp, dis_loss_train_tmp, gen_loss_train_tmp = 0, 0, 0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator.train()  # Set the model to train mode. This will enable dropout and so on if implemented
    discriminator.train()
    
    for batch_idx, imgs in enumerate(train_dataloader):

        # Configure model input
        imgs_lr = imgs["lr"].to(SR_config.DEVICE)
        imgs_hr = imgs["hr"].to(SR_config.DEVICE)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Adversarial ground truths
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        fake_hr = generator(imgs_lr)
        p_real = discriminator(imgs_hr)
        p_fake = discriminator(fake_hr.detach())
        dis_loss = compute_discriminator_loss(p_real, p_fake, loss_fn_dict)
        
        dis_train_loss += dis_loss.item() / len(train_dataloader)
        dis_loss_train_tmp += dis_loss.item()
        
        optimizer_D.zero_grad()

        dis_scaler.scale(dis_loss).backward()
        dis_scaler.step(optimizer_D)
        dis_scaler.update()

        # -----------------
        #  Train Generator
        # -----------------

        # with torch.cuda.amp.autocast(dtype=torch.float16):
        p_fake = discriminator(fake_hr)
        gen_loss = compute_generator_loss(p_fake, imgs_hr, fake_hr, loss_fn_dict, epoch, gen_train_loss_dict, len(train_dataloader))

        gen_train_loss += gen_loss.item() / len(train_dataloader)
        gen_loss_train_tmp += gen_loss.item()

        optimizer_G.zero_grad()
        
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(optimizer_G)
        gen_scaler.update()
        
        loss_count_train_tmp += 1
        
        if (batch_idx+1) % 10 == 0:
            print(f'Epoch {epoch+1}, batch no. {batch_idx+1}, gen. loss: {gen_loss_train_tmp/loss_count_train_tmp}, disc. loss: {dis_loss_train_tmp/loss_count_train_tmp}')
            if log_wandb:
                wandb.log({"train_loss/gen_loss": gen_loss_train_tmp/loss_count_train_tmp, "train_loss/disc_loss": dis_loss_train_tmp/loss_count_train_tmp})
                loss_count_train_tmp = 0
                gen_loss_train_tmp = 0
                dis_loss_train_tmp = 0

    if epoch % SR_config.VALIDATION_FREQUENCY == 0:
        print(f'Testing Epoch {epoch+1}')
        
        dis_valid_loss, gen_valid_loss = 0, 0
        dis_guess_real_valid, dis_guess_fake_valid = 0, 0
        psnr_valid, ssim_valid = 0, 0
        loss_count_test_tmp, dis_loss_test_tmp, gen_loss_test_tmp = 0, 0, 0
        
        with torch.inference_mode():
            generator.eval()
            discriminator.eval()

            for batch_idx, imgs in enumerate(test_dataloader):
                
                imgs_hr = imgs["hr"].to(SR_config.DEVICE)
                imgs_lr = imgs["lr"].to(SR_config.DEVICE)
                
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Generate fake high resolution images
                fake_hr = generator(imgs_lr)
                p_real = discriminator(imgs_hr)
                p_fake = discriminator(fake_hr.detach())
                dis_loss = compute_discriminator_loss(p_real, p_fake, loss_fn_dict)
                    
                # Compute average epoch loss
                dis_valid_loss += dis_loss.item() / len(test_dataloader)
                
                # Compute average discriminator probability on fake and real
                dis_guess_real_valid += (torch.sum(sigmoid_func(p_real))).item() / len(test_dataloader.dataset)
                dis_guess_fake_valid += (torch.sum(sigmoid_func(p_fake))).item() / len(test_dataloader.dataset)
                
                psnr, ssim = loss_functions.performance_metrics(imgs_hr, fake_hr)
                psnr_valid += psnr.item() / len(test_dataloader.dataset)
                ssim_valid += ssim.item() / len(test_dataloader.dataset)
                
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                prop_fake = discriminator(fake_hr)  # Compute prop_fake again, as the value was discarded by backward
                gen_loss = compute_generator_loss(prop_fake, imgs_hr, fake_hr, loss_fn_dict, epoch, gen_valid_loss_dict, len(test_dataloader))

                # Compute average epoch loss
                gen_valid_loss += gen_loss.item() / len(test_dataloader)
                
                gen_loss_test_tmp += gen_loss.item()
                dis_loss_test_tmp += dis_loss.item()
                loss_count_test_tmp += 1
                
                if (batch_idx+1) % 10 == 0:
                    if log_wandb:
                        wandb.log({"val_loss/gen_loss": gen_loss_test_tmp/loss_count_test_tmp, "val_loss/disc_loss": dis_loss_test_tmp/loss_count_test_tmp})
                        loss_count_test_tmp = 0
                        gen_loss_test_tmp = 0
                        dis_loss_test_tmp = 0
                if batch_idx+1 == len(test_dataloader)-1: 
                    if log_wandb:
                        img_grid = utils.make_images_from_vol(imgs_hr[-5:], imgs_lr[-5:], fake_hr[-5:])
                        image = wandb.Image(img_grid, caption="LR - HR - SR - dif")
                        wandb.log({"images": image})
    
    print(f"Discriminator training/validation loss in epoch {epoch+1}/{n_epochs} was {dis_train_loss:.4f}/{dis_valid_loss:.4f}")
    print(f"Generator GAN training/validation loss in epoch {epoch+1}/{n_epochs} was {gen_train_loss:.4f}/{gen_valid_loss:.4f}")
    print(f"Average PSNR of validation set in epoch {epoch + 1}/{n_epochs} was {psnr_valid:.4f}")
    print(f"Average SSIM of validation set in epoch {epoch + 1}/{n_epochs} was {ssim_valid:.4f}")
    print(f"Average discriminator guess on reals in epoch {epoch + 1}/{n_epochs} was {dis_guess_real_valid:.4f}")
    print(f"Average discriminator guess on fakes in epoch {epoch + 1}/{n_epochs} was {dis_guess_fake_valid:.4f}")

