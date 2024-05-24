## Training script for SRGAN model

# %% Packages
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import TrainData, TestData
from models import Discriminator, SRGenerator

# %% Settings
random.seed(42)     # For reproducability

use_amp = True      # For speeding up training loop
save_models = True

femur_train = "001"
femur_test = "002"

channels = 1
hr_height = 128
hr_width = 128
hr_shape = (hr_height, hr_width)
num_res_units = 4

n_epochs = 4
lr = 1e-4
batch_train = 12
batch_test = 4
coeff_GAN = 1e-3
coeff_pixel = 1
coeff_cont = 6e-3

opt_adam_b1 = 0.9 
opt_adam_b2 = 0.999 

root_path = "/dtu/3d-imaging-center/courses/02509/groups/members/soeba/"

dataset_path = root_path + "Data/"

go_cuda = torch.cuda.is_available()

# %% Make dataset paths
hr_path_train = dataset_path + "femur_" + femur_train + "/micro/f_" + femur_train + ".npy"
lr_path_train = dataset_path + "femur_" + femur_train + "/clinical/f_" + femur_train + ".npy"
mask_path_train = dataset_path + "femur_" + femur_train + "/mask/f_" + femur_train + ".npy"
img_paths_train = [hr_path_train, lr_path_train, mask_path_train]

hr_path_test = dataset_path + "femur_" + femur_test + "/micro/f_" + femur_test + ".npy"
lr_path_test = dataset_path + "femur_" + femur_test + "/clinical/f_" + femur_test + ".npy"
img_paths_test = [hr_path_test, lr_path_test]

# %% Datasets and dataloaders
train_dataloader = DataLoader(TrainData(file_paths=img_paths_train, patch_size=hr_height), batch_size=batch_train, drop_last=True, shuffle=True, num_workers=4)
test_dataloader = DataLoader(TestData(img_paths_test), batch_size=batch_test, drop_last=True, shuffle=True, num_workers=4)

# %% Train settings
# Initialize generator and discriminator
generator = SRGenerator(in_channels=channels, out_channels=1, n_residual_blocks=num_res_units)
discriminator = Discriminator(input_shape=(channels, *hr_shape))

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_pixel = torch.nn.L1Loss()

# Send to GPU
if go_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_pixel = criterion_pixel.cuda()
Tensor = torch.cuda.FloatTensor if go_cuda else torch.Tensor

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(opt_adam_b1, opt_adam_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(opt_adam_b1, opt_adam_b2))

scaler = torch.cuda.amp.GradScaler()        # For optimizing training loop

# %% Train and test loop
for epoch in range(n_epochs):
    ### Training
    print(f'Training Epoch {epoch+1}')
    
    torch.cuda.empty_cache()

    gen_loss, disc_loss, gen_count, disc_count = 0, 0, 0, 0
    for batch_idx, imgs in enumerate(train_dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Generate SR image
            imgs_sr = generator(imgs_lr)

            # Pixel loss
            loss_pixel = criterion_pixel(imgs_sr, imgs_hr)
            
            # Adversarial loss
            pred_fake = discriminator(imgs_sr)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

            # Total generator loss
            loss_G = coeff_pixel * loss_pixel + coeff_GAN * loss_GAN
        
        # Generator update
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()
        optimizer_G.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # Generate SR image
            imgs_sr = generator(imgs_lr)
            
            # Adversarial loss
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(imgs_sr.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))  
        
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
        torch.save(generator.state_dict(), root_path + "Model/model.pt")
        print(f'Saved model')
    
    # Print average loss after each epoch
    print(f'Epoch {epoch+1}, batch no. {batch_idx+1}, gen. loss: {gen_loss/gen_count}, disc. loss: {disc_loss/disc_count}')
    
    ### Testing
    print(f'Testing Epoch {epoch+1}')
    for batch_idx, imgs in enumerate(test_dataloader):
        with torch.inference_mode():
            
            generator.eval()

            # Configure model input
            imgs_lr = imgs["lr"].type(Tensor)
            imgs_hr = imgs["hr"].type(Tensor)

            # Generate SR images
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                imgs_sr = generator(imgs_lr)
                
            # TO-DO:
            # Her kan I tilføje hvad I nu har lyst til at teste, fx beregning af Bone Volume Fraction (BV/TV) eller Local Thickness...
            
            # Til at starte med kan i fx lave et plot med et HR, LR og SR slice
            if batch_idx == 10:
                fig, ax = plt.subplots(1,3,figsize=(16,4))
                ax[0].imshow(imgs_hr[0][0].cpu().detach(),cmap='gray')
                ax[0].axis('off')
                ax[1].imshow(imgs_lr[0][0].cpu().detach(),cmap='gray')
                ax[1].axis('off')
                ax[2].imshow(imgs_sr[0][0].cpu().detach(),cmap='gray')
                ax[2].axis('off')
                plt.show()
                plt.savefig(root_path + "Figures/test_figure.png")
                



# %%
