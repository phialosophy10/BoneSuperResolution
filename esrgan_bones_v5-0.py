## Ver. 5.0: UCSF data
## Also added content loss coefficient

# %% Packages
import numpy as np
import pandas as pd
import os, math, sys
import random
import glob
import utils
import wandb
import cv2

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from datasets import PatchData
from models import FeatureExtractor, Discriminator2D, SRGenerator, ESRGenerator
from loss_functions import VGGLoss

import matplotlib.pyplot as plt

import porespy as ps
ps.visualization.set_mpl_style()

from PIL import Image

from datetime import datetime

random.seed(42)

# %% Settings

# Logging with 'weights and biases'
log_wandb = True

if log_wandb:
    wandb.init(project="esrgan_bones_5.0", entity="phialosophy", config={})

# Variable settings
bone_no = ["SP02-02","SP02-03","SP02-04","SP02-05"]            # choose which bone(s) to train on
bone_no_test = "SP02-01"                                       # choose which bone to test on

n_epochs = 5
batch_size = 12
lr = 0.0001
lambda_adv = 1e-3
lambda_pixel = 1e-2
lambda_cont = 6e-3
warmup_batches = 600

if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])
if len(sys.argv) > 2:
    batch_size = int(sys.argv[2])
if len(sys.argv) > 3:
    lr = float(sys.argv[3])
if len(sys.argv) > 4:
    lambda_adv = float(sys.argv[4])
if len(sys.argv) > 5:
    lambda_pixel = float(sys.argv[5])
if len(sys.argv) > 6:
    lambda_cont = float(sys.argv[6])
if len(sys.argv) > 7:
    warmup_batches = int(sys.argv[7])
if len(sys.argv) > 8:
    bone_no_test = sys.argv[8]
if len(sys.argv) > 9:
    bone_no = []
    for i in range(9, len(sys.argv)):
        bone_no.append(sys.argv[i])

if log_wandb:
    DT = datetime.now()
    wandb.run.name = "train" + str(len(bone_no)) + "_" + str(n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")

channels = 1
hr_height = 128         # 512
hr_width = 128          # 512
hr_shape = (hr_height, hr_width)

dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/UCSF_data/"

# adam: decay of first order momentum of gradient
b1 = 0.9
# adam: decay of second order momentum of gradient
b2 = 0.999

go_cuda = torch.cuda.is_available()

if log_wandb:
    wandb.config.update({
        "learning_rate": lr,
        "epochs": n_epochs,
        "warmup_batches": warmup_batches,
        "batch_size": batch_size,
        "bone_train": bone_no,
        "bone_test": bone_no_test,
        "loss_coefficients (adv, pix, cont)": (lambda_adv, lambda_pixel, lambda_cont),
    })

# wandb.run.save()

# %% Make dictionaries with dataset paths
hr_paths_train, lr_paths_train = [], []
for i in range(len(bone_no)):
    hr_paths_train += sorted(glob.glob(dataset_path + bone_no[i] + "/mct/patches/" + bone_no[i] + "*.*")) #add "_???0" before "*.*"" to only include every 10'th slice
    lr_paths_train += sorted(glob.glob(dataset_path + bone_no[i] + "/XCT/patches/" + bone_no[i] + "*.*"))

hr_paths_test = sorted(glob.glob(dataset_path + bone_no_test + "/mct/patches/" + bone_no_test + "*.*"))
lr_paths_test = sorted(glob.glob(dataset_path + bone_no_test + "/XCT/patches/" + bone_no_test + "*.*"))

img_paths_train = []
img_paths_test = []
for i in range(len(hr_paths_train)):
    img_paths_train.append([hr_paths_train[i], lr_paths_train[i]])
for i in range(len(hr_paths_test)):
    img_paths_test.append([hr_paths_test[i], lr_paths_test[i]])
# %% Train and test loaders
train_dataloader = DataLoader(PatchData(img_paths_train, hr_shape=hr_shape), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=1)
test_dataloader = DataLoader(PatchData(img_paths_test, hr_shape=hr_shape), batch_size=batch_size, drop_last=False, shuffle=True, num_workers=1)

# %% Train model 

# Initialize generator and discriminator
generator = ESRGenerator(channels=1, filters=64, num_res_blocks=4) 
#generator = SRGenerator(in_channels=1, out_channels=, num_res_blocks=16)
discriminator = Discriminator2D(input_shape=(1, *hr_shape))

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = VGGLoss(layer_idx=35)
criterion_pixel = torch.nn.L1Loss()

if go_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    criterion_pixel = criterion_pixel.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if go_cuda else torch.Tensor

for epoch in range(n_epochs):
    print(f'Training Epoch {epoch+1}')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    gen_loss, disc_loss, gen_count, disc_count = 0, 0, 0, 0
    for batch_idx, imgs in enumerate(train_dataloader):

        batches_done = epoch * len(train_dataloader) + batch_idx

        # Configure model input
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)
        
        # -----------------
        #  Train Generator
        # -----------------
        
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        # Content loss
        loss_content = criterion_content(imgs_hr, gen_hr)
            
        if batches_done < warmup_batches:
            # Warm-up
            loss_G = lambda_pixel * loss_pixel + lambda_cont * loss_content
            loss_G.backward()
            optimizer_G.step()
            gen_loss += loss_G.item()
            if (batch_idx+1) % 10 == 0:
                if log_wandb:
                    wandb.log({"loss/gen_loss": gen_loss/gen_count})
            gen_count += 1
            continue
            
        # with torch.cuda.amp.autocast(dtype=torch.float16):
            # Extract validity predictions from discriminator
        pred_fake = discriminator(gen_hr)

        # Adversarial loss
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        # Total generator loss
        loss_G = lambda_pixel * loss_pixel + lambda_cont * loss_content + lambda_adv * loss_GAN
        
        optimizer_G.zero_grad()

        loss_G.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # with torch.cuda.amp.autocast(dtype=torch.float16):
        gen_hr = generator(imgs_lr)
        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images 
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real) - 0.1 * torch.ones_like(pred_real))  
    
        # Total loss
        loss_D = loss_real + loss_fake
        
        optimizer_D.zero_grad()

        loss_D.backward()
        optimizer_D.step()

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
        gen_count += 1
        disc_count += 1
        
        if (batch_idx+1) % 10 == 0:
            print(f'Epoch {epoch+1}, batch no. {batch_idx+1}, gen. loss: {gen_loss/gen_count}, disc. loss: {disc_loss/disc_count}')
            if log_wandb:
                wandb.log({"loss/gen_loss": gen_loss/gen_count, "loss/disc_loss": disc_loss/disc_count})
                #wandb.log({"loss/gen_loss": loss_G.item(), "loss/disc_loss": loss_D.item()})

    ### Testing
    print(f'Testing Epoch {epoch+1}')
    for batch_idx, imgs in enumerate(test_dataloader):
        with torch.inference_mode():
            # Save image grid with inputs and SRGAN outputs
            if batch_idx+1 == len(test_dataloader)-1: 

                generator.eval()

                # Configure model input
                imgs_lr = imgs["lr"].type(Tensor)
                imgs_hr = imgs["hr"].type(Tensor)

                # Generate a high resolution image from low resolution input
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                gen_hr = generator(imgs_lr)

                if log_wandb:
                    img_grid1, img_grid2 = utils.make_thickness_images_dif2(imgs_hr[-5:], imgs_lr[-5:], gen_hr[-5:])
                    image1 = wandb.Image(img_grid1, caption="HR - LR - SR - dif")
                    wandb.log({"images": image1})
                    image2 = wandb.Image(img_grid2, caption="thick HR - hist HR - thick SR - hist SR")
                    wandb.log({"thickness and histograms": image2})

                if epoch == n_epochs-1:
                    slice_ids = imgs["slice_name"]
                    for i in range(len(imgs_lr)):
                        im_np = gen_hr[i][0].cpu().detach()
                        np.save("/work3/soeba/HALOS/Results/SR/patches/f_" + slice_ids[i], im_np)
                        im_uint8 = np.uint8(im_np * 255)
                        cv2.imwrite("/work3/soeba/HALOS/Results/SR/patches/f_" + slice_ids[i] + ".png", im_uint8)
