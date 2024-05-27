## Packages
import sys
import os
import random
import torch
import numpy as np
import torchio as tio
from torch.utils.data import Dataset
# from torchvision.transforms import v2
import torchvision.transforms as transforms

## Dataset classes
class BlockData(Dataset):
    def __init__(self, files, shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # tio.ZNormalization()
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # tio.ZNormalization()
            ]
        )
        self.files = files
        self._height = shape[0]
        self._width = shape[1]
        self._depth = shape[2]

    def __getitem__(self, index):
        img_hr = np.load(self.files[index][0])
        img_lr = np.load(self.files[index][1])
        img_hr = self.hr_transform(img_hr)
        img_lr = self.lr_transform(img_lr)
        img_hr = torch.unsqueeze(img_hr, dim=0)                # add channel dimension
        img_lr = torch.unsqueeze(img_lr, dim=0)                # add channel dimension
        block_name = self.files[index][0]
        block_name = block_name[-12:-4]             # Remove ".npy" - consists of femur no. and block no. in the form 'FFF_BBBB'

        return {"lr": img_lr, "hr": img_hr, "block_name": block_name}       

    def __len__(self):
        return len(self.files)
    
class SliceData(Dataset):
    def __init__(self, file_paths):
        self.slice_nos = []
        self.hr_vols = []
        self.lr_vols = []
        self.vol_names = []
        self.slice_idx = []
        for i in range(len(file_paths)):
            hr_volume = np.load(file_paths[i][0])
            self.hr_vols.append(hr_volume)
            self.lr_vols.append(np.load(file_paths[i][1]))
            self.slice_nos.append(hr_volume.shape[2])
            self.vol_names.append(file_paths[i][0][64:67])          #[-7:-4]: "f_FFF.npy", [-16:-13]: "f_FFF_centered.npy", [64:67]: all
            for j in range(hr_volume.shape[2]):
                self.slice_idx.append([i,j])

        self.num_slices_total = np.sum(n for n in self.slice_nos)
        
        self.lr_transform = transforms.Compose([transforms.ToTensor()])
        self.hr_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_slices_total

    def __getitem__(self, index):
        # Get a slice from both hr and lr volumes
        hr_slice = self.hr_vols[self.slice_idx[index][0]][:,:,self.slice_idx[index][1]]
        lr_slice = self.lr_vols[self.slice_idx[index][0]][:,:,self.slice_idx[index][1]]

        slice_name = self.vol_names[self.slice_idx[index][0]] + "_" + str(self.slice_idx[index][1]).zfill(4)

        # Convert to PyTorch tensors
        hr_slice = self.hr_transform(hr_slice)
        lr_slice = self.lr_transform(lr_slice)

        # Return the data in the required format
        return {"lr": lr_slice, "hr": hr_slice, "slice_name": slice_name}
    
class SliceData_v2(Dataset):
    def __init__(self, file_paths, pad=True, num_patches=6):
        self.hr_vols = []
        self.lr_vols = []
        self.slice_idx = []
        self.hr_means = []
        self.hr_stds = []
        self.lr_means = []
        self.lr_stds = []
        num_bones = len(file_paths)
        num_patch_bone, rem = divmod(num_patches, num_bones)
        num_patch_bone_last = num_patch_bone + rem
        num_patch = [num_patch_bone] * (num_bones-1)
        num_patch.append(num_patch_bone_last)
        for i in range(num_bones):
            hr_vol = np.load(file_paths[i][0])
            self.hr_means.append(np.mean(hr_vol))
            self.hr_stds.append(np.std(hr_vol))
            self.hr_vols.append(hr_vol)
            lr_vol = np.load(file_paths[i][1])
            self.lr_means.append(np.mean(lr_vol))
            self.lr_stds.append(np.std(lr_vol))
            self.lr_vols.append(lr_vol)
            for j in range(num_patch[i]):
                s_idx = random.randrange(hr_vol.shape[2])
                self.slice_idx.append([i,s_idx])

        self.num_slices_total = num_patches
        
        self.lr_transforms = []
        self.hr_transforms = []
        for i in range(num_bones):
            self.lr_transforms.append(transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.lr_means[i], std=self.lr_stds[i])
            ]))
            self.hr_transforms.append(transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.hr_means[i], std=self.hr_stds[i])
            ]))

    def __len__(self):
        return self.num_slices_total

    def __getitem__(self, index):
        # Get a slice from both hr and lr volumes
        hr_slice = self.hr_vols[self.slice_idx[index][0]][:,:,self.slice_idx[index][1]]
        lr_slice = self.lr_vols[self.slice_idx[index][0]][:,:,self.slice_idx[index][1]]
        
        if pad:
            rows, cols = hr_slice.shape
            if rows <= 1024:
                x_pad = 1024
            else:
                x_pad = 2048
            if cols <= 1024:
                y_pad = 1024
            else:
                y_pad = 2048
            padded_slice_hr = np.zeros((x_pad, y_pad), dtype=hr_slice.dtype)
            padded_slice_lr = np.zeros((x_pad, y_pad), dtype=lr_slice.dtype)
            start_row = (x_pad - rows) // 2
            start_col = (y_pad - cols) // 2
            padded_slice_hr[start_row:start_row + rows, start_col:start_col + cols] = hr_slice
            padded_slice_lr[start_row:start_row + rows, start_col:start_col + cols] = lr_slice
            hr_slice = padded_slice_hr
            lr_slice = padded_slice_lr
        
        # Convert to PyTorch tensors
        hr_slice = self.hr_transforms[slice_idx[index][0]](hr_slice)
        lr_slice = self.lr_transforms[slice_idx[index][0]](lr_slice)

        # Return the data in the required format
        return {"lr": lr_slice, "hr": hr_slice}

class PatchData(Dataset):
    def __init__(self, files, hr_shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose([transforms.ToTensor()])
        self.hr_transform = transforms.Compose([transforms.ToTensor()])
        self.files = files

    def __getitem__(self, index):
        # img = Image.open(self.files[index % len(self.files)])
        img_hr = np.load(self.files[index][0])      # Image.open(self.files[index][0])
        img_lr = np.load(self.files[index][1])      # Image.open(self.files[index][1])
        img_hr = self.hr_transform(img_hr)
        img_lr = self.lr_transform(img_lr)
        slice_name = self.files[index][0]
        slice_name = slice_name[-15:-4]             # Remove ".npy" - consists of femur no., slice no. and patch no. in the form 'FFF_SSSS_PP'

        return {"lr": img_lr, "hr": img_hr, "slice_name": slice_name}       # "femur_name": femur_name, "slice_id": slice_id, "patch_no": patch_no

    def __len__(self):
        return len(self.files)

class PatchData_v2(Dataset):
    def __init__(self, file_paths, patch_size):
        self.hr_patches = []
        self.lr_patches = []
        num_pix_for_patch = 800
        patch_count = 0
        for i in range(len(file_paths)):
            hr_vol = np.load(file_paths[i][0])
            lr_vol = np.load(file_paths[i][1])
            mask = np.load(file_paths[i][2])
            block_x = hr_vol.shape[0]//patch_size
            block_y = hr_vol.shape[1]//patch_size
            for j in range(mask.shape[2]):
                im_hr = hr_vol[:,:,j]
                im_lr = lr_vol[:,:,j]
                im_m = mask[:,:,j]
                for k in range(block_x):
                    for l in range(block_y):
                        if np.sum(im_m[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size]) > num_pix_for_patch:
                            self.hr_patches.append(im_hr[k*patch_size:(k+1)*patch_size,l*patch_size:(l+1)*patch_size])
                            self.lr_patches.append(im_lr[k*patch_size:(k+1)*patch_size,l*patch_size:(l+1)*patch_size])
                            patch_count += 1

        self.num_patches_total = patch_count
        
        self.lr_transform = transforms.Compose([transforms.ToTensor()])
        self.hr_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.num_patches_total

    def __getitem__(self, index):
        # Get a patch from both hr and lr volumes
        hr_patch = self.hr_patches[index]
        lr_patch = self.lr_patches[index]

        # Convert to PyTorch tensors
        hr_patch = self.hr_transform(hr_patch)
        lr_patch = self.lr_transform(lr_patch)

        # Return the data in the required format
        return {"lr": lr_patch, "hr": hr_patch}

class PatchData_v3(Dataset):
    def __init__(self, file_paths, patch_size, num_patches=200000):
        self.hr_patches = []
        self.lr_patches = []
        self.vol_idx = []
        num_pix_for_patch = int(0.05 * (patch_size ** 2))
        patch_count = 0
        num_bones = len(file_paths)
        num_patch_bone, rem = divmod(num_patches, num_bones)
        num_patch_bone_last = num_patch_bone + rem
        num_patch = [num_patch_bone] * (num_bones-1)
        num_patch.append(num_patch_bone_last)
        self.hr_means = []
        self.hr_stds = []
        self.lr_means = []
        self.lr_stds = []
        for i in range(num_bones):
            hr_vol = np.load(file_paths[i][0])
            self.hr_means.append(np.mean(hr_vol))
            self.hr_stds.append(np.std(hr_vol))
            lr_vol = np.load(file_paths[i][1])
            self.lr_means.append(np.mean(lr_vol))
            self.lr_stds.append(np.std(lr_vol))
            mask = np.load(file_paths[i][2])
            patch_count_bone = 0
            while patch_count_bone < num_patch[i]:
                slice_idx = random.randrange(hr_vol.shape[2])
                x_idx = random.randrange(hr_vol.shape[0]-patch_size)
                y_idx = random.randrange(hr_vol.shape[1]-patch_size)
                patch_mask = mask[x_idx:x_idx+patch_size,y_idx:y_idx+patch_size,slice_idx]
                if np.sum(patch_mask) > num_pix_for_patch:
                    self.hr_patches.append(hr_vol[x_idx:x_idx+patch_size,y_idx:y_idx+patch_size,slice_idx])
                    self.lr_patches.append(lr_vol[x_idx:x_idx+patch_size,y_idx:y_idx+patch_size,slice_idx])
                    patch_count_bone += 1
                    self.vol_idx.append(i)

        self.num_patches = num_patches
        
        self.lr_transforms = []
        self.hr_transforms = []
        for i in range(num_bones):
            self.lr_transforms.append(transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.lr_means[i], std=self.lr_stds[i])
            ]))
            self.hr_transforms.append(transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.hr_means[i], std=self.hr_stds[i])
            ]))

    def __len__(self):
        return self.num_patches

    def __getitem__(self, index):
        # Get a patch from both hr and lr volumes
        hr_patch = self.hr_patches[index]
        lr_patch = self.lr_patches[index]

        # Convert to PyTorch tensors
        hr_patch = self.hr_transforms[vol_idx[index]](hr_patch)
        lr_patch = self.lr_transforms[vol_idx[index]](lr_patch)

        # Return the data in the required format
        return {"lr": lr_patch, "hr": hr_patch}
    
class PatchData_synth(Dataset):
    def __init__(self, files, hr_shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose([
                transforms.GaussianBlur(kernel_size=9, sigma=0.1),
                transforms.ToTensor()])
        self.hr_transform = transforms.Compose([transforms.ToTensor()])
        self.files = files

    def __getitem__(self, index):
        # img = Image.open(self.files[index % len(self.files)])
        img_hr = np.load(self.files[index][0])      # Image.open(self.files[index][0])
        img_lr = np.load(self.files[index][1])      # Image.open(self.files[index][1])
        img_hr = self.hr_transform(img_hr)
        img_lr = self.lr_transform(img_lr)
        slice_name = self.files[index][0]
        slice_name = slice_name[-15:-4]             # Remove ".npy" - consists of femur no., slice no. and patch no. in the form 'FFF_SSSS_PP'

        return {"lr": img_lr, "hr": img_hr, "slice_name": slice_name}       # "femur_name": femur_name, "slice_id": slice_id, "patch_no": patch_no

    def __len__(self):
        return len(self.files)