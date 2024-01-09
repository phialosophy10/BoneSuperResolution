# Packages
import glob
import os, math, sys
import utils
import cv2

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from torchvision.models import vgg19

# 3D models
class MultiLevelDenseNet(nn.Module):
    def __init__(self, in_c, k_factor=12, k_size=3):
        super().__init__()

        self.conv0 = nn.Conv3d(in_c, 2 * k_factor, k_size, stride=1, padding=1)

        self.dense_block0 = DenseBlock(2 * k_factor, k_factor, k_size)
        self.dense_block1 = DenseBlock(2 * k_factor, k_factor, k_size)
        self.dense_block2 = DenseBlock(2 * k_factor, k_factor, k_size)
        self.dense_block3 = DenseBlock(2 * k_factor, k_factor, k_size)

        self.compress0 = nn.Conv3d(8 * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)
        self.compress1 = nn.Conv3d(14 * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)
        self.compress2 = nn.Conv3d(20 * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)

        # Reconstruction via bottleneck from paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
        self.bottleneck = nn.Conv3d(26 * k_factor, 8 * k_factor, kernel_size=1, stride=1, padding=0)
        # Batch norm operation here?
        # self.SR0 = SRBlock3D(8 * k_factor, 8 * k_factor, 6, 2)  # In paper on densenets, they use 8*k channels for in and output
        # Final reconstruction with kernel size 3
        self.recon = nn.Conv3d(8 * k_factor, 1, kernel_size=k_size, stride=1, padding=1)

    def forward(self, input):
        # Initial convolution with 2k output filters
        x = self.conv0(input)

        z = self.dense_block0(x)
        skip = torch.cat([x, z], 1)
        comp = self.compress0(skip)

        z = self.dense_block1(comp)
        skip = torch.cat([skip, z], 1)
        comp = self.compress1(skip)

        z = self.dense_block2(comp)
        skip = torch.cat([skip, z], 1)
        comp = self.compress2(skip)

        z = self.dense_block3(comp)
        final_skip = torch.cat([skip, z], 1)

        z = self.bottleneck(final_skip)
        # z = self.SR0(z)

        out = self.recon(z)

        return out

class DenseBlock(nn.Module):
    def __init__(self, in_c, k_factor=12, k_size=3):
        super().__init__()

        # Four or seven dense blocks
        self.dense_unit0 = DenseUnit(in_c, k_factor, k_size)
        self.dense_unit1 = DenseUnit(in_c + k_factor, k_factor, k_size)
        self.dense_unit2 = DenseUnit(in_c + 2*k_factor, k_factor, k_size)
        self.dense_unit3 = DenseUnit(in_c + 3*k_factor, k_factor, k_size)

    def forward(self, input):
        x0 = self.dense_unit0(input)
        skip0 = torch.cat([input, x0], 1)

        x1 = self.dense_unit1(skip0)
        skip1 = torch.cat([skip0, x1], 1)

        x2 = self.dense_unit2(skip1)
        skip2 = torch.cat([skip1, x2], 1)

        x3 = self.dense_unit3(skip2)
        out = torch.cat([skip2, x3], 1)

        return out

class DenseUnit(nn.Module):
    def __init__(self, in_c, k_factor=12, k_size=3):
        super().__init__()

        self.norm0 = nn.BatchNorm3d(num_features=in_c)
        self.act0 = nn.ELU(alpha=1.0)
        self.conv0 = nn.Conv3d(in_c, k_factor, kernel_size=k_size, stride=1, padding=1)

    def forward(self, x):
        return self.conv0(self.act0(self.norm0(x)))

class SRBlock3D(nn.Module):
    def __init__(self, in_c, n, k_size=6, pad=2):
        super().__init__()

        #self.deconv0 = nn.ConvTranspose3d(in_c, n, kernel_size=k_size, stride=2, padding=pad, bias=0)
        self.deconv0 = nn.ConvTranspose3d(in_c, n, kernel_size=k_size, stride=2, padding=pad)
        self.act0 = nn.PReLU(num_parameters=n)

        # ICNR is an initialization method for sub-pixel convolution which removes checkerboarding
        # From the paper: Checkerboard artifact free sub-pixel convolution.
        # https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
        # weight = ICNR(self.deconv0.weight, initializer=nn.init.normal_, upscale_factor=2, mean=0.0, std=0.02)
        # self.deconv0.weight.data.copy_(weight)

    def forward(self, x):
        # Taken from this paper, rightmost method in figure 3: https://arxiv.org/pdf/1812.11440.pdf
        x = self.deconv0(x)
        out = self.act0(x)

        return out

class Discriminator3D(nn.Module):
    def __init__(self, input_size, in_c, n_conv_vec=[64,64,128,128,256,256,512,512], n_dense=[1024, 1], k_size=3):
        super().__init__()

        self.conv0 = nn.Conv3d(in_c, n_conv_vec[0], kernel_size=k_size, padding=1, stride=1)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        dim = [int(np.ceil(input_size/2**i)) for i in range(1,5)]

        self.LRconv0 = LRconvBlock3D(dim[0], n_conv_vec[0], n_conv_vec[1], k_size, stride=2)
        self.LRconv1 = LRconvBlock3D(dim[0], n_conv_vec[1], n_conv_vec[2], k_size, stride=1)
        self.LRconv2 = LRconvBlock3D(dim[1], n_conv_vec[2], n_conv_vec[3], k_size, stride=2)
        self.LRconv3 = LRconvBlock3D(dim[1], n_conv_vec[3], n_conv_vec[4], k_size, stride=1)
        self.LRconv4 = LRconvBlock3D(dim[2], n_conv_vec[4], n_conv_vec[5], k_size, stride=2)
        self.LRconv5 = LRconvBlock3D(dim[2], n_conv_vec[5], n_conv_vec[6], k_size, stride=1)
        self.LRconv6 = LRconvBlock3D(dim[3], n_conv_vec[6], n_conv_vec[7], k_size, stride=2)

        self.flatten = nn.Flatten()
        ll_size = int(n_conv_vec[7] * dim[3]**3)
        self.dense0 = nn.Linear(ll_size, n_dense[0])
        self.act1 = nn.LeakyReLU(0.2, inplace=True)  
        self.dense1 = nn.Linear(n_dense[0], n_dense[1])

        self.act_sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Initial convolution and LeakyRelu activation
        x = self.act0(self.conv0(input))

        # LeakyRelu convolution block network
        x = self.LRconv0(x)
        x = self.LRconv1(x)
        x = self.LRconv2(x)
        x = self.LRconv3(x)
        x = self.LRconv4(x)
        x = self.LRconv5(x)
        x = self.LRconv6(x)

        # Dense block network + LeakyRelu
        x = self.dense0(self.flatten(x))
        x = self.act1(x)
        out = self.dense1(x)

        # Final sigmoid activation (Remember to remove if BCEWithLogitsLoss() is used in training loop)
        #out = self.act_sigmoid(out)

        return out

class LRconvBlock3D(nn.Module):
    def __init__(self, input_size, in_c, n, k_size, stride):
        super().__init__()

        self.conv0 = nn.Conv3d(in_c, n, kernel_size=k_size, stride=stride, padding=1)
        #self.norm0 = nn.LayerNorm([DCSRN_config.BATCH_SIZE, n, 16, 16, 16])
        self.norm0 = nn.LayerNorm(input_size)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        out = self.act0(self.norm0(x))

        return out


## 2D models
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_35 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_35(torch.cat((img,img,img),1))
    
## SRGAN
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

class SRGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16): #16 #20
        super(SRGenerator, self).__init__()

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

## ESRGAN
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

class ESRGenerator(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=4, num_upsample=2):
        super(ESRGenerator, self).__init__()

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

class Discriminator2D(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator2D, self).__init__()

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