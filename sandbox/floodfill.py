# %%
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Read image

# %% Load a volume
path = "/work3/soeba/HALOS/Data/"

femur_no = "01" #"01", "15", "21", "74"

part = "a" #"a", "b"

vol_m = nib.load(path+"microCT/femur" + femur_no + "_mod.nii") #01, 15, 21, 74
vol_m = np.array(vol_m.dataobj)

vol_c_l = nib.load(path+"clinicalCT/resliced_femur" + femur_no + "_paper_linear.nii") #01, 15, 21, 74
vol_c_l = np.array(vol_c_l.dataobj)

x_dim, y_dim = 128, 128

vol_m = vol_m[:,:,200:1175]
vol_c_l = vol_c_l[:,:,200:1175]

block_x = vol_m.shape[0]//x_dim #4
block_y = vol_m.shape[1]//y_dim #5

# %% Select slice
show_ind = 200
im_m = vol_m[:,:,show_ind]
im_m_min = np.min(im_m)
im_m_max = np.max(im_m)

im_m = (im_m-im_m_min)/(im_m_max-im_m_min)

im_in = np.uint8(im_m * 255)

#im_in = Image.fromarray(np.uint8(im_m * 255)).convert("L")
#im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.imshow(im_in,cmap='gray')
plt.title("Image slice")
plt.show()

# %% Flood-fill 
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.
 
th, im_th = cv2.threshold(im_in, 60, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(im_th,cmap='gray')
plt.title("Thresholded Image")
plt.show()
 
# Copy the thresholded image.
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255)
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv
 
# Display images.
plt.figure()
plt.imshow(im_floodfill,cmap='gray')
plt.title("Floodfilled Image")
plt.show()
plt.figure()
plt.imshow(im_floodfill_inv,cmap='gray')
plt.title("Inverted Floodfilled Image")
plt.show()
plt.figure()
plt.imshow(im_out,cmap='gray')
plt.title("Foreground")
plt.show()
# %%
