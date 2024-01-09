## Ver. 3.1: Cleaned up code, moved model and dataset definitions to own .py files
## Also removed deprecated "Variable" function
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
from models import FeatureExtractor, Discriminator2D, GeneratorRRDB

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
    wandb.init(project="esrgan_bones_3.0", entity="phialosophy", config={})

# Variable settings
femur_no = ["002","013","015"]              # [001,002,013,015,021,026,074,075,083,086,138,164,172]      # choose which bone(s) to train on
femur_no_test = "001"                                            # choose which bone to test on
in_type = "clinical"            # "clinical", "micro"           # choose whether input in clinical CT or micro CT
out_type = "micro"              # "micro", "thick"              # choose whether target is micro CT or thickness image

n_epochs = 5
batch_size = 12
lr = 0.0001
lambda_adv = 5e-3
lambda_pixel = 1e-2
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
    warmup_batches = int(sys.argv[6])
if len(sys.argv) > 7:
    femur_no_test = sys.argv[7]
if len(sys.argv) > 8:
    femur_no = []
    for i in range(8, len(sys.argv)):
        femur_no.append(sys.argv[i])

if log_wandb:
    DT = datetime.now()
    wandb.run.name = "f" + femur_no[0] + "-" + femur_no_test + "_" + in_type + "-" + out_type + "_" + str(n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")

channels = 1
hr_height = 128         # 512
hr_width = 128          # 512
hr_shape = (hr_height, hr_width)

dataset_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"

# adam: decay of first order momentum of gradient
b1 = 0.9
# adam: decay of second order momentum of gradient
b2 = 0.999

cuda = torch.cuda.is_available()

if log_wandb:
    wandb.config.update({
        "learning_rate": lr,
        "epochs": n_epochs,
        "warmup_batches": warmup_batches,
        "batch_size": batch_size,
        "femur_train": femur_no,
        "femur_test": femur_no_test,
        "input_type": in_type,
        "loss_coefficients (adv, pix)": (lambda_adv, lambda_pixel),
        "output_type": out_type
    })

# wandb.run.save()

# %% Make dictionaries with dataset paths

hr_paths_train, lr_paths_train = [], []
for i in range(len(femur_no)):
    hr_paths_train += sorted(glob.glob(dataset_path + "femur_" + femur_no[i] + "/" + out_type + "/patches/f_" + femur_no[i] + "*.*"))
    lr_paths_train += sorted(glob.glob(dataset_path + "femur_" + femur_no[i] + "/" + in_type + "/patches/f_" + femur_no[i] + "*.*"))

hr_paths_test = sorted(glob.glob(dataset_path + "femur_" + femur_no_test + "/" + out_type + "/patches/f_" + femur_no_test + "*.*"))
lr_paths_test = sorted(glob.glob(dataset_path + "femur_" + femur_no_test + "/" + in_type + "/patches/f_" + femur_no_test + "*.*"))

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
generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=8)         #num_res_blocks=16
discriminator = Discriminator2D(input_shape=(1, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    criterion_pixel = criterion_pixel.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

for epoch in range(n_epochs):
    print(f'Training Epoch {epoch+1}')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    gen_loss, disc_loss, loss_count = 0, 0, 0
    for i, imgs in enumerate(train_dataloader):

        batches_done = epoch * len(train_dataloader) + i

        # Configure model input
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        with torch.cuda.amp.autocast(dtype=torch.float16):
            gen_hr = generator(imgs_lr)
            valid = Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape)))
            fake = Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape)))
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
        
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        
        optimizer_D.zero_grad()

        loss_D.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = lambda_adv * loss_GAN #+ lambda_pixel * loss_pixel + loss_content
        
        optimizer_G.zero_grad()

        loss_G.backward()
        optimizer_G.step()

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
        loss_count += 1
        
        print(f'Epoch {epoch+1}, batch no. {i+1}, gen. loss: {loss_G.item()}, disc. loss: {loss_D.item()}')

        if log_wandb:
            wandb.log({"loss/gen_loss": gen_loss/loss_count, "loss/disc_loss": disc_loss/loss_count})
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

                if out_type == "micro":
                    img_grid1, img_grid2 = utils.make_thickness_images_dif2(imgs_hr[-5:], imgs_lr[-5:], gen_hr[-5:])
                else:
                    img_grid1 = utils.make_images2(imgs_hr[:5], imgs_lr[:5], gen_hr[:5])
                if log_wandb:
                    image1 = wandb.Image(img_grid1, caption="LR - HR - SR - dif")
                    wandb.log({"images": image1})
                    if out_type == "micro":
                        image2 = wandb.Image(img_grid2, caption="thick LR - hist LR - thick SR - hist SR")
                        wandb.log({"thickness and histograms": image2})

            if epoch == n_epochs-1:

                generator.eval()

                # Configure model input
                imgs_lr = imgs["lr"].type(Tensor)
                slice_ids = imgs["slice_name"]

                # Generate a high resolution image from low resolution input
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    gen_hr = generator(imgs_lr)

                for i in range(len(imgs_lr)):
                    im_np = gen_hr[i][0].cpu().detach()
                    np.save("/work3/soeba/HALOS/Results/SR/patches/f_" + slice_ids[i], im_np)
                    im_uint8 = np.uint8(im_np * 255)
                    cv2.imwrite("/work3/soeba/HALOS/Results/SR/patches/f_" + slice_ids[i] + ".png", im_uint8)
