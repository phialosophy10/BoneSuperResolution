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
    wandb.init(project="esrgan_bones_gramgan", entity="phialosophy", config={})

# Variable settings
femur_no = ["15","21","74"] #"01", "15", "21", "74"        # choose which bone(s) to train on
femur_no_test = "01"                                       # choose which bone to test on 
in_type = "clinical" #"clinical", "micro"                  # choose whether input in clinical CT or micro CT
out_type = "micro" #"micro", "thick"                       # choose whether target is micro CT or thickness image
if in_type == "clinical":
    in_type += "/low-res/" #"/low-res/", "/hi-res"         # choose whether to train on low-res or hi-res clinical CT scans
    if in_type == "clinical/low-res/":
        in_type += "linear" #"nn", "linear"                # add interpolation type to string (linear/nearest-neighbor)

n_epochs = 5
batch_size = 4
lr = 0.0001
lambda_adv = 5e-3
lambda_pixel = 1e-2
warmup_batches = 600

if len(sys.argv)>1:
    n_epochs = int(sys.argv[1])
if len(sys.argv)>2:
    batch_size = int(sys.argv[2])
if len(sys.argv)>3:
    lr = float(sys.argv[3])
if len(sys.argv)>4:
    lambda_adv = float(sys.argv[4])
if len(sys.argv)>5:
    lambda_pixel = float(sys.argv[5])
if len(sys.argv)>6:
    warmup_batches = int(sys.argv[6])
if len(sys.argv)>7:
    femur_no_test = sys.argv[7]
if len(sys.argv)>8:
    femur_no = []
    for i in range(8,len(sys.argv)):
        femur_no.append(sys.argv[i])

DT = datetime.now()
wandb.run.name = "f" + femur_no[0] + "-" + femur_no_test + "_" + in_type + "-" + out_type + "_" + str(n_epochs) + "epochs_" + DT.strftime("%d/%m_%H:%M")

channels = 1
hr_height = 128 #512
hr_width = 128 #512
hr_shape = (hr_height, hr_width)

dataset_path = "/work3/soeba/HALOS/Data/Images"

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

#wandb.run.save()

# %% Make dictionaries with dataset paths

hr_paths_train, lr_paths_train = [], []
for i in range(len(femur_no)):
    hr_paths_train += sorted(glob.glob(dataset_path + "/" + out_type + "/" + femur_no[i] + "*.*"))
    lr_paths_train += sorted(glob.glob(dataset_path + "/" + in_type + "/" + femur_no[i] + "*.*"))

hr_paths_test = sorted(glob.glob(dataset_path + "/" + out_type + "/" + femur_no_test + "*.*"))
lr_paths_test = sorted(glob.glob(dataset_path + "/" + in_type + "/" + femur_no_test + "*.*"))

img_paths_train = []
img_paths_test = []
for i in range(len(hr_paths_train)):
    img_paths_train.append([hr_paths_train[i], lr_paths_train[i]])
for i in range(len(hr_paths_test)):
    img_paths_test.append([hr_paths_test[i], lr_paths_test[i]])

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
        slice_id = self.files[index][0]
        slice_id = slice_id[-12:-4]

        return {"lr": img_lr, "hr": img_hr, "slice_id": slice_id}

    def __len__(self):
        return len(self.files)

# %% Train and test loaders

train_dataloader = DataLoader(ImageDataset(img_paths_train, hr_shape=hr_shape), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=1)
test_dataloader = DataLoader(ImageDataset(img_paths_test, hr_shape=hr_shape), batch_size=batch_size, drop_last=False, shuffle=True, num_workers=1)

# %% Define Model Classes

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         vgg19_model = vgg19(pretrained=True)
#         self.vgg19_35 = nn.Sequential(*list(vgg19_model.features.children())[:35])

#     def forward(self, img):
#         return self.vgg19_35(torch.cat((img,img,img),1))

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1), #nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1)
                nn.LeakyReLU(),
               # nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
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
        # for i in range(len(layers)):
        #     self.layer[i] = layers[i]

    def forward(self, img):
        return self.model(img)

# %% Train model 

# Initialize generator and discriminator
generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=16)
discriminator = Discriminator(input_shape=(1, *hr_shape))
#feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
#feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

def style_loss(feat_real, feat_fake):
    def gram_matrix(t):
        einsum = torch.einsum('bijc,bijd->bcd', t, t)
        n_pix = list(t.size())[1]*list(t.size())[2]
        return einsum/n_pix

    real_gram_mat = []
    fake_gram_mat = []

    # all D features except logits
    for i in range(len(feat_real)):
        real_gram_mat.append(gram_matrix(feat_real[i]))
        fake_gram_mat.append(gram_matrix(feat_fake[i]))

    # l1 loss
    style_loss = torch.sum(torch.stack([torch.mean(torch.abs(real_gram_mat[idx] - fake_gram_mat[idx]))
                           for idx in range(len(feat_real))]))

    return style_loss / len(feat_real)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
  #  feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    criterion_pixel = criterion_pixel.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

for epoch in range(n_epochs):
    print(f'Training Epoch {epoch+1}')

    gen_loss, disc_loss, loss_count = 0, 0, 0
    for i, imgs in enumerate(train_dataloader):

        batches_done = epoch * len(train_dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G_loss pixel: %f]"
                % (epoch, n_epochs, i+1, len(train_dataloader), loss_pixel.item())
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        feature_extractor = discriminator.children()
        gen_features, real_features = [], []
        for child in feature_extractor:
            gen_features.append(child(gen_hr))
            real_features.append(child(imgs_hr))
       # gen_features = feature_extractor(gen_hr)
       # real_features = feature_extractor(imgs_hr).detach()
       # loss_content = criterion_content(gen_features, real_features)
        loss_content = style_loss(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
        loss_count += 1

        if log_wandb:
            wandb.log({"loss/gen_loss": gen_loss/loss_count, "loss/disc_loss": disc_loss/loss_count})
          #  wandb.log({"loss/gen_loss": loss_G.item(), "loss/disc_loss": loss_D.item()})


    ### Testing
    print(f'Testing Epoch {epoch+1}')
    for batch_idx, imgs in enumerate(test_dataloader):
    
        # Save image grid with inputs and SRGAN outputs
        if batch_idx+1 == len(test_dataloader)-1: 

            generator.eval()

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor), requires_grad=False)
            imgs_hr = Variable(imgs["hr"].type(Tensor), requires_grad=False)

            # Generate a high resolution image from low resolution input
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

        # if epoch == n_epochs-1:

        #     generator.eval()

        #     # Configure model input
        #     imgs_lr = Variable(imgs["lr"].type(Tensor), requires_grad=False)
        #     slice_ids = imgs["slice_id"]

        #     # Generate a high resolution image from low resolution input
        #     gen_hr = generator(imgs_lr)

        #     for i in range(len(imgs_lr)):
        #         im = Image.fromarray(np.uint8(gen_hr[i][0].cpu().detach()*255)).convert("L")
        #         im.save(dataset_path + "/SR" + in_type[8:] + "/" + slice_ids[i] + ".jpg")
