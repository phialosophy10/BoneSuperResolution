## Packages
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

## Dataset classes
class TestData(Dataset):
    def __init__(self, file_paths):
        self.slice_nos = []
        self.hr_vol = np.load(file_paths[0])
        self.lr_vol = np.load(file_paths[1])

        self.num_slices = self.hr_vol.shape[2]
        
        self.transform = transforms.Compose([
                transforms.ToTensor()])

    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        # Get a slice from both hr and lr volumes
        hr_slice = self.hr_vol[:,:,index]
        lr_slice = self.lr_vol[:,:,index]
        rows, cols = hr_slice.shape
        
        # Pad slices to size [1024x2048], so that the network can do convolutions and tranform to torch tensor
        padded_hr = np.zeros((1024, 2048), dtype=hr_slice.dtype)
        padded_lr = np.zeros((1024, 2048), dtype=lr_slice.dtype)
        start_row = (1024 - rows) // 2
        start_col = (2048 - cols) // 2
        padded_hr[start_row:start_row + rows, start_col:start_col + cols] = hr_slice
        padded_lr[start_row:start_row + rows, start_col:start_col + cols] = lr_slice
        hr_slice = self.transform(padded_hr)
        lr_slice = self.transform(padded_lr)

        # Return the data in the required format
        return {"lr": lr_slice, "hr": hr_slice}
    
class TrainData(Dataset):
    def __init__(self, file_paths, patch_size):
        self.hr_patches = []
        self.lr_patches = []
        num_pix_for_patch = 800             #... og denne variabel
        patch_count = 0
        hr_vol = np.load(file_paths[0])
        lr_vol = np.load(file_paths[1])
        mask = np.load(file_paths[2])
        block_x = hr_vol.shape[0]//patch_size
        block_y = hr_vol.shape[1]//patch_size
        for j in range(hr_vol.shape[2]):
            im_hr = hr_vol[:,:,j]
            im_lr = lr_vol[:,:,j]
            im_m = mask[:,:,j]              # Hvis I ikke har masker og ikke synes det giver mening at lave dem, så kan I bare fjerne denne linje...
            for k in range(block_x):
                for l in range(block_y):
                    if np.sum(im_m[k*patch_size:(k+1)*patch_size, l*patch_size:(l+1)*patch_size]) > num_pix_for_patch:  #... og dette if-statement...
                        self.hr_patches.append(im_hr[k*patch_size:(k+1)*patch_size,l*patch_size:(l+1)*patch_size])
                        self.lr_patches.append(im_lr[k*patch_size:(k+1)*patch_size,l*patch_size:(l+1)*patch_size])
                        patch_count += 1

        self.num_patches_total = patch_count
        
        self.lr_transform = transforms.Compose([
                transforms.ToTensor()])
        self.hr_transform = transforms.Compose([
                transforms.ToTensor()])

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