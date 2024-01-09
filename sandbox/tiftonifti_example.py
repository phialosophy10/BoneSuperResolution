## This script loads all tif-slices in a folder, stacks them into a volume and saves them as a NiFTI-file.
## You need to input the origin and voxel-spacings if these are known.

# Packages
import numpy as np
import nibabel as nib
from PIL import Image
import os

# Set path
path = "/path/to/files/"

# Search and load stack of images
listFiles = []
for img_files in os.listdir(path):
    if img_files.endswith(".tif"):
        listFiles.append(img_files)

# Get dimensions of dataset
nFiles = len(listFiles)
im = np.array(Image.open(path + listFiles[0]))
imgDim = [im.shape[0],im.shape[1],nFiles]

# Initialize array
vol = np.zeros((imgDim[0],imgDim[1],imgDim[2]),np.uint16) #np.uint8

# Load all files
for file in listFiles:
    id = int(file[-8:-4])   #Identify slice number, if this is part of the file name, e.g. "slice_0341.tif"
    vol[:,:,id] = np.array(Image.open(path + file))

# Simple header for volume
origin = np.array([0,0,0])  # Location of corner voxel, use (0,0,0) if nothing else is known from the scanner

affine = np.zeros((4,4))
affine[0:3,3] = origin
affine[0,0] = 0.0528 # spacing in x [mm]
affine[1,1] = 0.0528 # spacing in y [mm]
affine[2,2] = 0.0528 # spacing in z [mm]

# Save volume as NiFTI
niiVol = nib.Nifti1Image(vol, affine) # Where vol is a 3D matrix of the volume (numpy array)
nib.save(niiVol, path + "nre_file_name.nii") 
