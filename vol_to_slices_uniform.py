# %% Packages
import numpy as np
import matplotlib.pyplot as plt
import sys

# %% Define cropping function
def crop_slice(arr, arr_center, crop_size=1024, pad_value=-1000.0):
    """
    Crop a region of an arbitrarily sized 2D numpy array centered at a given index.
    Pad with zeros if the cropped region is beyond the boundary of the array.

    Parameters:
    - arr: 2D numpy array
    - arr_center: tuple, center index (row, column) for cropping
    - crop_size: int, size of the cropped region (default is 1024)
    - pad_value: value to pad cropped array with (default is -1000.0)

    Returns:
    - cropped_array: 2D numpy array, the cropped region
    """

    # Extracting array dimensions
    rows, cols = arr.shape

    # Calculate the starting and ending indices for the crop
    start_row = max(0, arr_center[0] - crop_size // 2)
    end_row = min(rows, arr_center[0] + crop_size // 2)

    start_col = max(0, arr_center[1] - crop_size // 2)
    end_col = min(cols, arr_center[1] + crop_size // 2)

    # Create an array of zeros with the desired crop size
    cropped_array = pad_value * np.ones((crop_size, crop_size), dtype=arr.dtype)

    # Calculate the region to copy from the original array
    orig_start_row = max(0, crop_size // 2 - arr_center[0])
    orig_end_row = min(crop_size, orig_start_row + end_row - start_row)

    orig_start_col = max(0, crop_size // 2 - arr_center[1])
    orig_end_col = min(crop_size, orig_start_col + end_col - start_col)

    # Copy the region from the original array to the cropped array
    cropped_array[orig_start_row:orig_end_row, orig_start_col:orig_end_col] = arr[start_row:end_row, start_col:end_col]

    return cropped_array

## MANUAL
# %% Load volume and mask
# crop_size = 1024
# femur_no = '172'
# res = 'clinical'
# vol = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur_no + '/' + res + '/volume/f_' + femur_no + '.npy')
# mask = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur_no + '/mask/volume/f_' + femur_no + '.npy')

# # %% Choose slice
# slice_no = 1400
# im = vol[:,:,slice_no]
# im_mask = mask[:,:,slice_no]

# # %% Calculate center and crop image around it
# slice_centers = [np.argmax(np.sum(im_mask,axis=1)), np.argmax(np.sum(im_mask,axis=0))]
# cropped_im = crop_slice(im,slice_centers,crop_size=crop_size)

# # %% Show the cropped slice
# plt.imshow(cropped_im,cmap='gray')
# plt.plot(512.0, 512.0, 'ro')
# plt.show()

# # %% Show original slice and center
# plt.imshow(im,cmap='gray')
# plt.plot(slice_centers[1], slice_centers[0], 'ro')
# plt.show()

## AUTOMATED
# %% Choose volumes, loop through and get centers from masks, center slices of both volumes and save

crop_size = 1024

femur_no = ['172']
if len(sys.argv) > 1:
    femur_no = []
    for i in range(1, len(sys.argv)):
        femur_no.append(sys.argv[i])

for k in range(len(femur_no)):
    mask = np.load('/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur_no[k] + '/mask/volume/f_' + femur_no[k] + '.npy')
    slice_centers = []
    for j in range(mask.shape[2]):
        slice_centers.append([np.argmax(np.sum(mask[:,:,j],axis=1)), np.argmax(np.sum(mask[:,:,j],axis=0))])
    for res in ['clinical', 'micro']:
        path = '/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/femur_' + femur_no[k] + '/' + res + '/volume/'
        vol = np.load(path + 'f_' + femur_no[k] + '.npy')
        cropped_vol = np.zeros((crop_size,crop_size,vol.shape[2]))
        for m in range(vol.shape[2]):
            cropped_vol[:,:,m] = crop_slice(vol[:,:,m],slice_centers[m],crop_size=crop_size,pad_value=-1000.0).astype(np.float32)
        np.save(path + 'f_' + femur_no[k] + '_centered.npy', cropped_vol)

# %%
