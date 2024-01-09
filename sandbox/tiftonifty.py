# %% Import packages
import numpy as np
import nibabel as nib
from PIL import Image
import skimage.io as skio
import os

# %% Load volume
# vol_tmp = Image.open('/work3/soeba/HALOS/Data/microCT/femur74.tiff') #01 15 21 74
# vol = np.array(vol_tmp)

#dataFolder = "/work3/soeba/HALOS/Data/microCT/femur_15_merged_tiff/"
dataFolder = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_02/tiff_uint8/"

# Searching and loading stack of images
listFiles = []
for img_files in os.listdir(dataFolder):
    if img_files.endswith(".tif"):
        listFiles.append(img_files)

# Get dimensions of dataset
nFiles = len(listFiles)
im = np.array(Image.open(dataFolder + listFiles[0]))
imgDim = [im.shape[0],im.shape[1],nFiles]

# Initialize array
vol = np.zeros((imgDim[0],imgDim[1],imgDim[2]),np.uint16) #np.uint8

# Load all files:
for file in listFiles:
    id = int(file[-8:-4]) #Identify slice number
    vol[:,:,id] = np.array(Image.open( dataFolder + file ))

# %% Location of corner voxel:
#origin = np.array([7.109,-99.39,-818]) # Use (0,0,0) if nothing else is known from the scanner
origin = np.array([0,0,0])

# %% Simple header for volume
affine = np.zeros((4,4))
affine[0:3,3] = origin
affine[0,0] = 0.0528 # spacing in x [mm]
affine[1,1] = 0.0528 # spacing in y [mm]
affine[2,2] = 0.0528 # spacing in z [mm]

# %% Make nii variable
niiVol = nib.Nifti1Image(vol, affine) #Where vol is a 3D matrix of the volume (numpy array)

# %% Save as:
nib.save(niiVol, '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_02/femur_02.nii') #01 15 21 74

# %%
