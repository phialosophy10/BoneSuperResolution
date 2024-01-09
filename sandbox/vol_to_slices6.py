## Loads a volume and creates 128x128 pixel slices. It throws away empty slices
## In this script we only save low-res/linear CT images
## Sandbox for different segmentation methods, incl. using local thickness, deformable snakes and blurring/smoothing

# %% Packages
import nibabel as nib
import skimage.io
from skimage.morphology import erosion, dilation
import numpy as np
import porespy as ps
import cv2
from IPython import display
import pylab as pl
import time
ps.visualization.set_mpl_style()

import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter

import simple_snake as sis

# %% Load a volume
path = "/work3/soeba/HALOS/Data/"

femur_no = "01" #"01", "15", "21", "74"

part = "a" #"a", "b"

vol_m = nib.load(path+"microCT/femur" + femur_no + "_mod.nii") #01, 15, 21, 74
vol_m = np.array(vol_m.dataobj)

vol_c_l = nib.load(path+"clinicalCT/resliced_femur" + femur_no + "_paper_linear.nii") #01, 15, 21, 74
vol_c_l = np.array(vol_c_l.dataobj)

x_dim, y_dim = 128, 128

# %% Select ROI
if femur_no == "01":
    vol_m = vol_m[:,:,200:1175]
    vol_c_l = vol_c_l[:,:,200:1175]

    block_x = vol_m.shape[0]//x_dim #4
    block_y = vol_m.shape[1]//y_dim #5

elif femur_no == "15":
    if part == "a":
        vol_m = vol_m[:,:,150:675]
        vol_m = vol_m.transpose((1, 0, 2))
        vol_c_l = vol_c_l[:,:,150:675]
        vol_c_l = vol_c_l.transpose((1, 0, 2))

    elif part == "b":
        vol_m = vol_m[:,:,676:1200]
        vol_m = vol_m.transpose((1, 0, 2))
        vol_c_l = vol_c_l[:,:,676:1200]
        vol_c_l = vol_c_l.transpose((1, 0, 2))

    block_x = vol_m.shape[0]//x_dim #4
    block_y = vol_m.shape[1]//y_dim #6

elif femur_no == "21":
    if part == "a":
        vol_m = vol_m[:,:,125:675]
        vol_c_l = vol_c_l[:,:,125:675]

    elif part == "b":
        vol_m = vol_m[:,:,676:1150]
        vol_c_l = vol_c_l[:,:,676:1150]

    block_x = vol_m.shape[0]//x_dim #5
    block_y = vol_m.shape[1]//y_dim #6

elif femur_no == "74":
    if part == "a":
        vol_m = vol_m[:,:,150:675]
        vol_c_l = vol_c_l[:,:,150:675]

    elif part == "b":
        vol_m = vol_m[:,:,676:1250]
        vol_c_l = vol_c_l[:,:,676:1250]

    block_x = vol_m.shape[0]//x_dim #4
    block_y = vol_m.shape[1]//y_dim #6

# %% Normalize
m_min = np.min(vol_m)
m_max = np.max(vol_m)

c_l_min = np.min(vol_c_l)
c_l_max = np.max(vol_c_l)

vol_m_norm = (vol_m-m_min)/(m_max-m_min)
vol_c_l_norm = (vol_c_l-c_l_min)/(c_l_max-c_l_min)

# %% Threshold for determining if slice is empty
if femur_no == "01":
    thresh = 0.15
elif femur_no == "15":
    thresh = 0.12
elif femur_no == "21":
    thresh = 0.18
elif femur_no == "74":
    thresh = 0.12

# %% Choose slice
show_ind = 150
im_m = vol_m_norm[:,:,show_ind] #np.uint8(vol_m_norm[:,:,show_ind] * 255)
#pore_sizes = list(np.linspace(1.0, 50, num=25))
#cort_thresh = 4 #threshold for thickness of cortical structure

# # %% Initialize snake
# nr_points = 100
# nr_iter = 60
# step_size = 60
# alpha = 0.5
# beta = 0.1

# center = np.array([250, 320]) #np.array((im_m.shape[0],im_m.shape[1]))/2
# radius = 220 #0.3*np.mean(im_m.shape)

# snake = sis.make_circular_snake(nr_points, center, radius)
# B = sis.regularization_matrix(nr_points, alpha, beta)

# plt.imshow(im_m, cmap=plt.cm.gray)
# plt.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
# plt.title('Initialization')
# plt.show()

# # %% Evolve snake
# for i in range(nr_iter):
#     snake = sis.evolve_snake(snake, im_m, B, step_size)
#     if i % 5 == 4:
#         print(f'Snake has been updated {i+1} times...')

# plt.imshow(im_m, cmap=plt.cm.gray)
# plt.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
# plt.title(f'Result after {nr_iter} iterations')
# plt.show()

# %% Create mask and fill inside of snake
# mask = np.zeros(im_m.shape)
# snake = tuple([np.transpose(snake)])
# cv2.fillPoly(mask,pts=tuple(snake),color=(255,255,255))

# plt.imshow(mask)

# %% Blur image with gaussian
im_blur = np.copy(im_m)

# Choose parameters:
alpha_blur = 25     #blur/sharpen parameter?
sigma_blur = 5      #size of gaussian kernel for blurring
sigma_sharp = 1     #Should be 1
blur_iters = 60     #number of times to blur and sharpen image
thresh_low = 0.175  #lower threshold (segments background from bone)
thresh_hi = 0.40    #upper threshold (semgents trabecular from cortical bone)
erosion_iters = 15  #number of times to erode the bone mask

for i in range(blur_iters):
    if i % 20 == 0:
        sigma_blur_temp=7
    else:
        sigma_blur_temp=sigma_blur
    im_blur = gaussian_filter(im_blur, sigma=sigma_blur_temp)
    filter_blur = gaussian_filter(im_blur, sigma=sigma_sharp)
    im_sharp = im_blur + alpha_blur * (im_blur - filter_blur)
    im_blur = im_sharp

plt.imshow(im_sharp, cmap='gray')
plt.title(f'Blurred, then sharpened image (repeated {blur_iters} times, with sigma={sigma_blur})')
plt.show()
# %%
mask = np.zeros(im_sharp.shape)
mask[im_blur > thresh_low] = 1

cross = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])

for i in range(erosion_iters):
    mask = erosion(mask, cross)

#mask[(mask == 1) & (im_blur < thresh_hi)] = 2

plt.imshow(mask)
plt.title('Mask')
plt.show()

masked_im_back = np.copy(im_m)
masked_im_back[mask != 0] = 0
masked_im_cort = np.copy(im_m)
masked_im_cort[mask != 1] = 0
# masked_im_trab = np.copy(im_m)
# masked_im_trab[mask != 2] = 0

fig, axs = plt.subplots(2, 1, sharey=True, figsize=(6,10))
axs[0].imshow(masked_im_back, cmap='gray')
axs[0].set_title('Background')
axs[1].imshow(masked_im_cort, cmap='gray')
axs[1].set_title('Bone')
# axs[2].imshow(masked_im_trab, cmap='gray')
# axs[2].set_title('Trabecular bone')
plt.suptitle('Masked images')
plt.show()

# %%
# masked_im = np.copy(im_m)
# masked_im[mask != 0] = 0
# plt.imshow(masked_im,cmap='gray')

# %% Initialize inner snake
# nr_points = 100
nr_iter = 100
step_size = 70
alpha = 0.1
# beta = 0.1

center_inner = np.array([250, 270]) #np.array((im_m.shape[0],im_m.shape[1]))/2
radius_inner = 65 #0.3*np.mean(im_m.shape)

snake_inner = sis.make_circular_snake(nr_points, center_inner, radius_inner)
B_inner = sis.regularization_matrix(nr_points, alpha, beta)

plt.imshow(im_sharp, cmap=plt.cm.gray)
plt.plot(np.r_[snake_inner[1],snake_inner[1,0]],np.r_[snake_inner[0],snake_inner[0,0]],'b-')
plt.title('Initialization')
plt.show()

# %% Evolve inner snake
for i in range(nr_iter):
    snake_inner = sis.evolve_snake(snake_inner, im_sharp, B_inner, step_size)
    if i % 5 == 4:
        print(f'Snake has been updated {i+1} times...')

plt.imshow(im_m, cmap=plt.cm.gray)
plt.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
plt.plot(np.r_[snake_inner[1],snake_inner[1,0]],np.r_[snake_inner[0],snake_inner[0,0]],'b-')
plt.title(f'Result after {nr_iter} iterations')
plt.show()

# %%
# plt.imshow(im_m,cmap='gray')
# plt.title('original image')
# plt.show()

# mask = np.copy(im_m)
# im_blur = np.copy(im_m)

# mask[mask <= 102] = 0
# mask[mask != 0] = 1

# plt.imshow(mask)
# plt.title('mask (bone/air)')
# plt.show()

# im_blur_tmp = np.copy(im_blur)
# im_blur = gaussian_filter(im_blur, sigma=7)
# im_blur[im_blur <= 50] = 0
# ## TO-DO: Find way to make sharp outer edge...
# im_blur[im_blur != 0] = 1
# mask = im_blur
# im_blur = im_blur_tmp

# plt.imshow(mask)
# plt.title('binary mask (bone/not-bone)')
# plt.show()

# plt.imshow(im_blur,cmap='gray')
# plt.title('blurred image')
# plt.show()

# mask[im_thk > cort_thresh] = 2

# plt.imshow(mask)
# plt.title('mask (background, trabecular, cortical)')
# plt.show()

# masked_im = np.copy(im_m)
# masked_im[mask == 0] = 0
# masked_im[mask == 2] = 0
# plt.imshow(masked_im,cmap='gray')
# plt.title('Trabecular structure masked')
# plt.show()

# %%
count = 0
mask = np.zeros((vol_m.shape[0],vol_m.shape[1]))
for i in range(vol_m.shape[2]):
    im_m = np.uint8(vol_m_norm[:,:,i] * 255)
    im_bin = np.copy(im_m)
    im_bin = gaussian_filter(im_bin, sigma=7)
    im_bin[im_bin <= 102] = 0  #threshold for segmenting bone/not-bone: 102
    im_bin[im_bin > 102] = 1
    for j in range(block_x):
        for k in range(block_y):
            if np.mean(vol_m_norm[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim,i]) > thresh:
                im_m = np.uint8(vol_m_norm[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim,i] * 255)
                np.save(path + "Images/micro/" + femur_no + part + "_" + str(count).zfill(4), im_m)
                im_c_l = np.uint8(vol_c_l_norm[j*x_dim:(j+1)*x_dim,k*y_dim:(k+1)*y_dim,i] * 255)
                np.save(path+"Images/clinical/low-res/linear/" + femur_no + part + "_" + str(count).zfill(4), im_c_l)
                count += 1

# %%
