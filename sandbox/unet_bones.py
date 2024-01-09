# %% Packages
import numpy as np
import pandas as pd
import os, math, sys
import random
import glob
import utils

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt

import monai
from monai.metrics import DiceMetric

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
    wandb.init(project="unet_bones", entity="phialosophy", config={})

# Variable settings
femur_no = "21" #"01", "15", "21", "74"                    # choose which bone to train on
femur_no_test = "74"                                       # choose which bone to test on
in_type = "clinical" #"clinical", "micro"                     # choose whether input in clinical CT or micro CT
out_type = "micro" #"micro", "thick"                       # choose whether target is micro CT or thickness image
if in_type == "clinical":
    in_type += "/low-res/" #"/low-res/", "/hi-res"            # choose whether to train on low-res or hi-res clinical CT scans
    if in_type == "clinical/low-res/":
        in_type += "linear" #"nn", "linear"               # add interpolation type to string (linear/nearest-neighbor)
n_epochs = 20
batch_size = 16 #8
lr = 1e-3 #0.00008
kernel_size = 5

DT = datetime.now()
wandb.run.name = "f" + femur_no + "-" + femur_no_test + "_" + in_type + "-" + out_type + "_" + str(n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")

channels = 1
hr_height = 128 #512
hr_width = 128 #512
hr_shape = (hr_height, hr_width)

dataset_path = "/work3/soeba/HALOS/Data/Images"

cuda = torch.cuda.is_available()

if log_wandb:
    wandb.config.update({
    "learning_rate": lr,
    "epochs": n_epochs,
    "batch_size": batch_size,
    "femur_train": femur_no,
    "femur_test": femur_no_test,
    "kernel_size": kernel_size,
    "input_type": in_type,
    "output_type": out_type
    })

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

else:
    # If training and testing on different femurs
    train_inds = list(range(len(hr_paths_train)))
    test_inds = list(range(len(hr_paths_test)))
    random.shuffle(test_inds)
    test_inds = test_inds[:int(len(test_inds)/5.0)]

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
test_dataloader = DataLoader(ImageDataset(img_paths_test, hr_shape=hr_shape), batch_size=int(batch_size*0.75), drop_last=True, shuffle=True, num_workers=1)

# %% Define Model Classes

# 2D UNet
model = utils.make_monai_net(layer_specs = (4, 8, 16, 32, 64, 128, 256), r_drop = 0.0, norm_type = 'instance', num_res = 2, kernel=kernel_size)

# %% Train model

# loss_function = monai.losses.DiceLoss(sigmoid=True)
loss_function = torch.nn.MSELoss()
# loss_function = torch.nn.L1Loss()

if cuda:
    model.cuda()
    loss_function.cuda()

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_losses = []
val_losses = []

for epoch in range(n_epochs):

    ### Training
    train_loss = 0
    print(f'Training Epoch {epoch}')
    for batch_idx, imgs in enumerate(train_dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        optimizer.zero_grad()
        outputs = model(imgs_lr)
        loss = loss_function(outputs, imgs_hr)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    ### Testing
    val_loss = 0
    print(f'Testing Epoch {epoch}')
    for batch_idx, imgs in enumerate(test_dataloader):

        model.eval()
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor), requires_grad=False)
        imgs_hr = Variable(imgs["hr"].type(Tensor), requires_grad=False)

        outputs_val = model(imgs_lr)
        loss_val = loss_function(outputs_val, imgs_hr)
        val_loss += loss_val.item()

        # Save image grid with inputs and SRGAN outputs
        if batch_idx+1 == len(test_dataloader)-1:
            if out_type == "micro":
                img_grid1, img_grid2 = utils.make_thickness_images_dif2(imgs_hr[:5], imgs_lr[:5], outputs_val[:5])
            else:
                img_grid1 = utils.make_images2(imgs_hr[:5], imgs_lr[:5], outputs_val[:5])
            if log_wandb:
                image1 = wandb.Image(img_grid1, caption="LR - HR - SR - dif")
                wandb.log({"images": image1})
                if out_type == "micro":
                    image2 = wandb.Image(img_grid2, caption="thick LR - hist LR - thick SR - hist SR")
                    wandb.log({"thickness and histograms": image2})

    train_loss = train_loss / len(train_dataloader)
    train_losses.append(train_loss)
    val_loss = val_loss / len(test_dataloader)
    val_losses.append(val_loss)

    if log_wandb:
        wandb.log({"loss/training loss": train_loss, "loss/validation loss": val_loss})

    print(f'Epoch: {epoch+1}/{n_epochs}.. Training loss: {train_loss}.. Validation Loss: {val_loss}')

    # Save model checkpoints
    if np.argmin(val_losses) == len(val_losses)-1:
        torch.save(model.state_dict(), "/work3/soeba/HALOS/saved_models/model.pth")