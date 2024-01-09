# %%
import skimage.io as skio
import numpy as np
import os
from PIL import Image

#%% Loading Data
# Select data folder with .tif
dataFolder = "/work3/soeba/HALOS/Data/microCT/femur_01_merged_tiff/"
#imgSpace = 3.6e-3

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
vol = np.zeros((imgDim[0],imgDim[1],imgDim[2]),np.uint16)

# Load all files:
for file in listFiles:
    id = int(file[-7:-4]) #Identify slice number
    vol[:,:,id] = np.array(Image.open( dataFolder + file ))

# %%
