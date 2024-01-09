## Loads a .npy volume and creates 128x128x128 blocks, which are also saved as .npy.
## We use the mask to discard (nearly) empty blocks.

# %% Packages
import numpy as np
import sys
import os

# %% Load a volume
root_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/"

femur_no = "026"                        # [001,002,013,015,021,026,031,074,075,083,086,138,164,172]
if len(sys.argv) > 1:
    femur_no = sys.argv[1]

root_path += "femur_" + femur_no + "/"

x_dim, y_dim, z_dim = 128, 128, 128      # Set dimension of blocks
num_vox_for_block = 5000                 # Threshold for determining if block is empty (or almost empty)

# %% Process the three volumes and save as blocks

res_list = ["mask", "clinical", "micro"]
dir_list = []
for res in res_list:
    dir_list.append(root_path + res + "/blocks_128/")
        
for direc in dir_list:
    if not os.path.exists(direc):
        os.makedirs(direc)

## Process mask volume
res = "mask/"
path = root_path + res
vol = np.load(path + "volume/f_" + femur_no + ".npy")

block_x = vol.shape[0]//x_dim           # Calculating the number of blocks to divide the volume into in the x-direction
block_y = vol.shape[1]//y_dim           # ...and in the y-direction
block_z = vol.shape[2]//z_dim           # ...and in the z-direction

# Make boolean array for registry of non-empty patches
full_blocks = np.full((block_x, block_y, block_z), False)

# Split into blocks and save
for i in range(block_z): 
    for j in range(block_x): 
        for k in range(block_y): 
            if np.sum(vol[j*x_dim:(j+1)*x_dim, k*y_dim:(k+1)*y_dim, i*z_dim:(i+1)*z_dim]) > num_vox_for_block:
                full_blocks[j,k,i] = True
                block_no = i*block_x*block_y+j*block_y+k
                block = vol[j*x_dim:(j+1)*x_dim, k*y_dim:(k+1)*y_dim, i*z_dim:(i+1)*z_dim]
                block = block.astype(np.float32)       # make correct datatype to be read by dataloader
                np.save(path + "blocks_128/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(block_no).zfill(4), block)

## Process CT volumes
for res in ["clinical/", "micro/"]:
    path = root_path + res
    vol = np.load(path + "volume/f_" + femur_no + ".npy")
    
    block_x = vol.shape[0]//x_dim           # Calculating the number of blocks to divide the volume into in the x-direction
    block_y = vol.shape[1]//y_dim           # ...and in the y-direction
    block_z = vol.shape[2]//z_dim           # ...and in the z-direction

    # Split into blocks and save
    for i in range(block_z): 
        for j in range(block_x): 
            for k in range(block_y):
                if full_blocks[j,k,i]:
                    block_no = i*block_x*block_y+j*block_y+k
                    block = vol[j*x_dim:(j+1)*x_dim, k*y_dim:(k+1)*y_dim, i*z_dim:(i+1)*z_dim]
                    block = block.astype(np.float32)       # make correct datatype to be read by dataloader
                    np.save(path + "blocks_128/f_" + femur_no + "_" + str(i).zfill(4) + "_" + str(block_no).zfill(4), block)
