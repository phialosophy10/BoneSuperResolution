## Packages
import sys
import os
import torch
import numpy as np
import torchio as tio
from torch.utils.data import Dataset
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
    
class PatchData(Dataset):
    def __init__(self, files, hr_shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose([
                transforms.ToTensor()])
        self.hr_transform = transforms.Compose([
                transforms.ToTensor()])
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
            self.vol_names.append(file_paths[i][0][-7:-4])
            for j in range(hr_volume.shape[2]):
                self.slice_idx.append([i,j])

        self.num_slices_total = np.sum(n for n in self.slice_nos)
        
        self.lr_transform = transforms.Compose([
                transforms.ToTensor()])
        self.hr_transform = transforms.Compose([
                transforms.ToTensor()])

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
    
class PatchData_v2(Dataset):
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
            self.vol_names.append(file_paths[i][0][-7:-4])
            for j in range(hr_volume.shape[2]):
                self.slice_idx.append([i,j])

        self.num_slices_total = np.sum(n for n in self.slice_nos)
        
        self.lr_transform = transforms.Compose([
                transforms.ToTensor()])
        self.hr_transform = transforms.Compose([
                transforms.ToTensor()])

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