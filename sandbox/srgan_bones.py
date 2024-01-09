# %% Packages 
import numpy as np
import pandas as pd
import os, math, sys
import random
import glob
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt

import porespy as ps
ps.visualization.set_mpl_style()

from PIL import Image

from datetime import datetime

import wandb

random.seed(42)

# %% Settings

# Logging with 'weights and biases'
log_wandb = True

if log_wandb:
    wandb.init(project="srgan_bones", entity="phialosophy", config={})

# Variable settings
femur_no = "21" #"01", "15", "21", "74"                    # choose which bone to train on
femur_no_test = "74"                                       # choose which bone to test on 
in_type = "clinical" #"clinical", "micro"                     # choose whether input in clinical CT or micro CT
out_type = "micro" #"micro", "thick"                       # choose whether target is micro CT or thickness image
if in_type == "clinical":
    in_type += "/low-res/" #"/low-res/", "/hi-res"            # choose whether to train on low-res or hi-res clinical CT scans
    if in_type == "clinical/low-res/":
        in_type += "linear" #"nn", "linear"               # add interpolation type to string (linear/nearest-neighbor)
n_epochs = 10
batch_size = 64 #8, 16
lr = 0.00008 #0.00016

DT = datetime.now()
wandb.run.name = "f" + femur_no + "-" + femur_no_test + "_" + in_type + "-" + out_type + "_" + str(n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")

channels = 1
hr_height = 128 #512
hr_width = 128 #512
hr_shape = (hr_height, hr_width)

dataset_path = "/work3/soeba/HALOS/Data/Images"

# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of second order momentum of gradient
b2 = 0.999
# epoch from which to start lr decay
decay_epoch = 15 #100

cuda = torch.cuda.is_available()

if log_wandb:
    wandb.config.update({
    "learning_rate": lr,
    "epochs": n_epochs,
    "batch_size": batch_size,
    "femur_train": femur_no,
    "femur_test": femur_no_test,
    "input_type": in_type,
    "output_type": out_type
    })

#wandb.run.save()

# %% Make dictionaries with dataset paths

hr_paths_train = sorted(glob.glob(dataset_path + "/" + out_type + "/" + femur_no + "*.*"))
hr_paths_test = sorted(glob.glob(dataset_path + "/" + out_type + "/" +femur_no_test + "*.*"))
lr_paths_train = sorted(glob.glob(dataset_path + "/" + in_type + "/" + femur_no + "*.*"))
lr_paths_test = sorted(glob.glob(dataset_path + "/" + in_type + "/" + femur_no_test + "*.*"))

# If training and testing on same femur
if femur_no == femur_no_test:
    n_ims = len(hr_paths_train)
    split_ind = int(n_ims/5.0)
    inds = list(range(n_ims))
    random.shuffle(inds)
    test_inds = inds[:split_ind]
    train_inds = inds[split_ind:]

    coeff_GAN = 1 #0.25 #1e-1
    coeff_cont = 0.5
else:
    # If training and testing on different femurs
    train_inds = list(range(len(hr_paths_train)))
    test_inds = list(range(len(hr_paths_test)))
    random.shuffle(test_inds)
    test_inds = test_inds[:int(len(test_inds)/5.0)]

    coeff_GAN = 1 #0.5
    coeff_cont = 0.5

img_paths_train = []
img_paths_test = []
for i in range(len(train_inds)):
    img_paths_train.append([hr_paths_train[train_inds[i]], lr_paths_train[train_inds[i]]])
for i in range(len(test_inds)):
    img_paths_test.append([hr_paths_test[test_inds[i]], lr_paths_test[test_inds[i]]])

# %% Dataset Class

class ImageDataset(Dataset):
    def __init__(self, files, hr_shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.files = files
    
    def __getitem__(self, index):
        # img = Image.open(self.files[index % len(self.files)])
        img_hr = np.load(self.files[index][0]) #Image.open(self.files[index][0])
        img_lr = np.load(self.files[index][1]) #Image.open(self.files[index][1])
        img_hr = self.hr_transform(img_hr)
        img_lr = self.lr_transform(img_lr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

# %% Train and test loaders

train_dataloader = DataLoader(ImageDataset(img_paths_train, hr_shape=hr_shape), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=1)
test_dataloader = DataLoader(ImageDataset(img_paths_test, hr_shape=hr_shape), batch_size=int(batch_size*0.25), drop_last=True, shuffle=True, num_workers=1)

# %% Define Model Classes

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(torch.cat((img,img,img),1))


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16): #16 #20
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

# %% Train model 

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_gen_losses, train_disc_losses, train_counter = [], [], []
test_gen_losses, test_disc_losses = [], []
test_counter = [idx*len(train_dataloader.dataset) for idx in range(1, n_epochs+1)]

for epoch in range(n_epochs):

    ### Training
    gen_loss, disc_loss = 0, 0
    print(f'Training Epoch {epoch}')
    for batch_idx, imgs in enumerate(train_dataloader):
        generator.train(); discriminator.train()
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        ### Train Generator
        optimizer_G.zero_grad()
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        loss_content += coeff_cont * criterion_content(gen_hr, imgs_hr)
        # Total loss
        loss_G = loss_content + coeff_GAN * loss_GAN #loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()

        ### Train Discriminator
        optimizer_D.zero_grad()
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        gen_loss += loss_G.item()
        train_gen_losses.append(loss_G.item())
        disc_loss += loss_D.item()
        train_disc_losses.append(loss_D.item())
        
        train_counter.append(batch_idx*batch_size + imgs_lr.size(0) + epoch*len(train_dataloader.dataset))
       # tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))
        print(f'Trained batch {batch_idx} in epoch {epoch}, gen_loss = {gen_loss/(batch_idx+1)}, disc_loss = {disc_loss/(batch_idx+1)} \n')
        if log_wandb:
            wandb.log({"loss/gen_loss": gen_loss/(batch_idx+1), "loss/disc_loss": disc_loss/(batch_idx+1)})

    ### Testing
    gen_loss, disc_loss = 0, 0
    print(f'Testing Epoch {epoch}')
    for batch_idx, imgs in enumerate(test_dataloader):
        generator.eval(); discriminator.eval()
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor), requires_grad=False)
        imgs_hr = Variable(imgs["hr"].type(Tensor), requires_grad=False)
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        ### Eval Generator
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        loss_content += coeff_cont * criterion_content(gen_hr, imgs_hr)
        # Total loss
        loss_G = loss_content + coeff_GAN * loss_GAN #loss_content + 1e-3 * loss_GAN

        ### Eval Discriminator
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
    
        # Save image grid with inputs and SRGAN outputs
        if batch_idx+1 == len(test_dataloader)-1: 
            # imgs_hr = make_grid(imgs_hr[:5], nrow=1, normalize=True)
            # gen_hr = make_grid(gen_hr[:5], nrow=1, normalize=True)
            # imgs_lr = make_grid(imgs_lr[:5], nrow=1, normalize=True)
            # img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
          # save_image(img_grid, f"/work3/soeba/HALOS/Data/figures/epoch{epoch}batch{batch_idx}.png", normalize=False)
            if out_type == "micro":
                img_grid1, img_grid2 = utils.make_thickness_images_dif2(imgs_hr[:5], imgs_lr[:5], gen_hr[:5])
            else:
                img_grid1 = utils.make_images2(imgs_hr[:5], imgs_lr[:5], gen_hr[:5])
            if log_wandb:
                image1 = wandb.Image(img_grid1, caption="LR - HR - SR - dif")
                wandb.log({"images": image1})
                if out_type == "micro":
                    image2 = wandb.Image(img_grid2, caption="thick LR - hist LR - thick SR - hist SR")
                    wandb.log({"thickness and histograms": image2})
            

    test_gen_losses.append(gen_loss/len(test_dataloader))
    test_disc_losses.append(disc_loss/len(test_dataloader))

    # Save model checkpoints
    if np.argmin(test_gen_losses) == len(test_gen_losses)-1:
        torch.save(generator.state_dict(), "/work3/soeba/HALOS/saved_models/generator.pth")
        torch.save(discriminator.state_dict(), "/work3/soeba/HALOS/saved_models/discriminator.pth")

    torch.cuda.empty_cache()