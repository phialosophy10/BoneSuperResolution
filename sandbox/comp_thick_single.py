# %% Packages
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
ps.visualization.set_mpl_style()
from PIL import Image
import glob
from numpy import cumsum
from numpy import abs
from numpy import sum
import sys

# %% Set data paths
femur_no = ["01", "15", "21", "74"]

in_type = "low-res/linear/" #"hi-res/" or "low-res/linear/" or "low-res/nn/"
dataset_path = "/work3/soeba/HALOS/Data/Images"
hr_paths, lr_paths, sr_paths, mask_paths = [], [], [], []
for i in range(len(femur_no)):
    hr_paths.append(sorted(glob.glob(dataset_path + "/micro/" + femur_no[i] + "*.*")))
    lr_paths.append(sorted(glob.glob(dataset_path + "/clinical/" + in_type + femur_no[i] + "*.*")))
    sr_paths.append(sorted(glob.glob(dataset_path + "/SR/" + in_type + femur_no[i] + "*.*")))
    mask_paths.append(sorted(glob.glob(dataset_path + "/masks/" + femur_no[i] + "*.*")))

# %% Show
# voxel_size= 0.12 * 1e-3 #voxel size is 120 micrometers (?)
# show_ind = 8000 #1001 #8000 #1503 #10504
# pore_sizes = list(np.linspace(1.0, 50, num=25))

# im_m = np.load(hr_paths[show_ind])
# im_c = np.load(lr_paths[show_ind])
# im_s = np.uint8(np.load(sr_paths[show_ind]) * 255)
# mask = np.load(mask_paths[show_ind])
# #im_s = np.asarray(Image.open(sr_paths[show_ind]))

# # Show images
# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
# axs[0].imshow(im_m,cmap='gray')
# axs[1].imshow(im_c,cmap='gray')
# axs[2].imshow(im_s,cmap='gray')
# plt.suptitle('Grayscale images')

# # Make binary
# bone_thresh = 75
# im_m[im_m<=bone_thresh] = 0
# im_m[im_m>bone_thresh] = 1

# im_c[im_c<=60] = 0
# im_c[im_c>60] = 1

# im_s[im_s<=bone_thresh] = 0
# im_s[im_s>bone_thresh] = 1

# # Show binary images
# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
# axs[0].imshow(im_m)
# axs[1].imshow(im_c)
# axs[2].imshow(im_s)
# plt.suptitle('Binary images (thickness)')

# # Make inverse binary images (for spacing calculations)
# im_m_sp = np.zeros(im_m.shape)
# im_m_sp[im_m==0] = 1
# im_m_sp[mask==0] = 0

# im_c_sp = np.zeros(im_c.shape)
# im_c_sp[im_c==0] = 1
# im_c_sp[mask==0] = 0

# im_s_sp = np.zeros(im_s.shape)
# im_s_sp[im_s==0] = 1
# im_s_sp[mask==0] = 0

# # Show inverse binary images
# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
# axs[0].imshow(im_m_sp)
# axs[1].imshow(im_c_sp)
# axs[2].imshow(im_s_sp)
# plt.suptitle('Binary images (spacing)')

# # Calculate local thickness images
# thk_m = ps.filters.local_thickness(im_m, sizes=pore_sizes, mode='dt')
# thk_c = ps.filters.local_thickness(im_c, sizes=pore_sizes, mode='dt')
# thk_s = ps.filters.local_thickness(im_s, sizes=pore_sizes, mode='dt')

# # Show thickness images
# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
# axs[0].imshow(thk_m)
# axs[1].imshow(thk_c)
# axs[2].imshow(thk_s)
# plt.suptitle('Thickness images')

# # Calculate local spacing image
# sp_m = ps.filters.local_thickness(im_m_sp, sizes=pore_sizes, mode='dt')
# sp_c = ps.filters.local_thickness(im_c_sp, sizes=pore_sizes, mode='dt')
# sp_s = ps.filters.local_thickness(im_s_sp, sizes=pore_sizes, mode='dt')

# # Show spacing images
# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
# axs[0].imshow(sp_m)
# axs[1].imshow(sp_c)
# axs[2].imshow(sp_s)
# plt.suptitle('Spacing images')

# # Calculate poresize distribution for thickness
# data_m = ps.metrics.pore_size_distribution(im=thk_m, bins=pore_sizes, log=False)
# data_c = ps.metrics.pore_size_distribution(im=thk_c, bins=pore_sizes, log=False)
# data_s = ps.metrics.pore_size_distribution(im=thk_s, bins=pore_sizes, log=False)

# # Plot histograms
# fig, ax = plt.subplots(1, 3, figsize=[15, 4])
# ax[0].bar(data_m.bin_centers*voxel_size, data_m.pdf, data_m.bin_widths*voxel_size, edgecolor='k')
# ax[1].bar(data_c.bin_centers*voxel_size, data_c.pdf, data_c.bin_widths*voxel_size, edgecolor='k')
# ax[2].bar(data_s.bin_centers*voxel_size, data_s.pdf, data_s.bin_widths*voxel_size, edgecolor='k')
# ax[0].set_title('Thickness, HR')
# ax[1].set_title('Thickness, LR')
# ax[2].set_title('Thickness, SR')
# plt.suptitle('Thickness histograms')

# # Calculate poresize distribution for spacing
# data_m_sp = ps.metrics.pore_size_distribution(im=sp_m, bins=pore_sizes, log=False)
# data_c_sp = ps.metrics.pore_size_distribution(im=sp_c, bins=pore_sizes, log=False)
# data_s_sp = ps.metrics.pore_size_distribution(im=sp_s, bins=pore_sizes, log=False)

# # Plot histograms
# fig, ax = plt.subplots(1, 3, figsize=[15, 4])
# ax[0].bar(data_m_sp.bin_centers*voxel_size, data_m_sp.pdf, data_m_sp.bin_widths*voxel_size, edgecolor='k')
# ax[1].bar(data_c_sp.bin_centers*voxel_size, data_c_sp.pdf, data_c_sp.bin_widths*voxel_size, edgecolor='k')
# ax[2].bar(data_s_sp.bin_centers*voxel_size, data_s_sp.pdf, data_s_sp.bin_widths*voxel_size, edgecolor='k')
# ax[0].set_title('Spacing, HR')
# ax[1].set_title('Spacing, LR')
# ax[2].set_title('Spacing, SR')
# plt.suptitle('Spacing histograms')

# # Mean local thickness and spacing
# mean_thk_m = np.sum(data_m.bin_centers*voxel_size*data_m.pdf)/np.sum(data_m.pdf)
# mean_thk_c = np.sum(data_c.bin_centers*voxel_size*data_c.pdf)/np.sum(data_c.pdf)
# mean_thk_s = np.sum(data_s.bin_centers*voxel_size*data_s.pdf)/np.sum(data_s.pdf)

# mean_sp_m = np.sum(data_m_sp.bin_centers*voxel_size*data_m_sp.pdf)/np.sum(data_m_sp.pdf)
# mean_sp_c = np.sum(data_c_sp.bin_centers*voxel_size*data_c_sp.pdf)/np.sum(data_c_sp.pdf)
# mean_sp_s = np.sum(data_s_sp.bin_centers*voxel_size*data_s_sp.pdf)/np.sum(data_s_sp.pdf)

# # Print
# print(f'Mean local thickness for HR volume: {mean_thk_m}\n')
# print(f'Mean local thickness for LR volume: {mean_thk_c}\n')
# print(f'Mean local thickness for SR volume: {mean_thk_s}\n')
# print(f'Mean local spacing for HR volume: {mean_sp_m}\n')
# print(f'Mean local spacing for LR volume: {mean_sp_c}\n')
# print(f'Mean local spacing for SR volume: {mean_sp_s}\n')

# %% Statistics
voxel_size= 0.12 * 1e-3 #voxel size is 120 micrometers (?)
bone_thresh = 75
n_patches = 20
start_slice = -500

pore_sizes = list(np.linspace(1.0, 50, num=25))

thickness_all, spacing_all = [], []
for j in range(len(femur_no)):
    thick_m, thick_c, thick_s, space_m, space_c, space_s = [], [], [], [], [], []
    for i in range(n_patches): 
        im_m = np.load(hr_paths[j][start_slice-i])
        im_c = np.load(lr_paths[j][start_slice-i])
        im_s = np.uint8(np.load(sr_paths[j][start_slice-i]) * 255)
        mask = np.load(mask_paths[j][start_slice-i])

        # Make binary
        im_m[im_m<=bone_thresh] = 0
        im_m[im_m>bone_thresh] = 1

        im_c[im_c<=60] = 0
        im_c[im_c>60] = 1

        im_s[im_s<=bone_thresh] = 0
        im_s[im_s>bone_thresh] = 1

        # Make inverse binary
        im_m_sp = np.zeros(im_m.shape)
        im_m_sp[im_m==0] = 1
        im_m_sp[mask==0] = 0

        im_c_sp = np.zeros(im_c.shape)
        im_c_sp[im_c==0] = 1
        im_c_sp[mask==0] = 0

        im_s_sp = np.zeros(im_s.shape)
        im_s_sp[im_s==0] = 1
        im_s_sp[mask==0] = 0

        # Calculate local thickness images
        thk_m = ps.filters.local_thickness(im_m, sizes=pore_sizes, mode='dt')
        thk_c = ps.filters.local_thickness(im_c, sizes=pore_sizes, mode='dt')
        thk_s = ps.filters.local_thickness(im_s, sizes=pore_sizes, mode='dt')

        # Calculate local spacing images
        sp_m = ps.filters.local_thickness(im_m_sp, sizes=pore_sizes, mode='dt')
        sp_c = ps.filters.local_thickness(im_c_sp, sizes=pore_sizes, mode='dt')
        sp_s = ps.filters.local_thickness(im_s_sp, sizes=pore_sizes, mode='dt')

        # Calculate poresize distribution for thickness
        data_m = ps.metrics.pore_size_distribution(im=thk_m, bins=pore_sizes, log=False)
        data_c = ps.metrics.pore_size_distribution(im=thk_c, bins=pore_sizes, log=False)
        data_s = ps.metrics.pore_size_distribution(im=thk_s, bins=pore_sizes, log=False)

        # Add to mean thickness
        if not np.isnan(np.sum(data_m.pdf)):
            thick_m.append(np.sum(data_m.bin_centers*voxel_size*data_m.pdf)/np.sum(data_m.pdf))
        if not np.isnan(np.sum(data_c.pdf)):
            thick_c.append(np.sum(data_c.bin_centers*voxel_size*data_c.pdf)/np.sum(data_c.pdf))
        if not np.isnan(np.sum(data_s.pdf)):
            thick_s.append(np.sum(data_s.bin_centers*voxel_size*data_s.pdf)/np.sum(data_s.pdf))

        # Calculate poresize distribution for thickness
        data_m_sp = ps.metrics.pore_size_distribution(im=sp_m, bins=pore_sizes, log=False)
        data_c_sp = ps.metrics.pore_size_distribution(im=sp_c, bins=pore_sizes, log=False)
        data_s_sp = ps.metrics.pore_size_distribution(im=sp_s, bins=pore_sizes, log=False)

        # Add to mean spacing
        if not np.isnan(np.sum(data_m_sp.pdf)):
            space_m.append(np.sum(data_m_sp.bin_centers*voxel_size*data_m_sp.pdf)/np.sum(data_m_sp.pdf))
        if not np.isnan(np.sum(data_c_sp.pdf)):
            space_c.append(np.sum(data_c_sp.bin_centers*voxel_size*data_c_sp.pdf)/np.sum(data_c_sp.pdf))
        if not np.isnan(np.sum(data_s_sp.pdf)):
            space_s.append(np.sum(data_s_sp.bin_centers*voxel_size*data_s_sp.pdf)/np.sum(data_s_sp.pdf))
            
    thickness_all.append([thick_m, thick_c, thick_s])
    spacing_all.append([space_m, space_c, space_s])

# %% Plot thickness
plt.figure()
plt.plot(thickness_all[0][0], thickness_all[0][2], 'b+')
plt.plot(thickness_all[1][0], thickness_all[1][2], 'g+')
plt.plot(thickness_all[2][0], thickness_all[2][2], 'r+')
plt.plot(thickness_all[3][0], thickness_all[3][2], 'y+')
plt.plot(np.mean(thickness_all[0][0]), np.mean(thickness_all[0][2]), 'bo')
plt.plot(np.mean(thickness_all[1][0]), np.mean(thickness_all[1][2]), 'go')
plt.plot(np.mean(thickness_all[2][0]), np.mean(thickness_all[2][2]), 'ro')
plt.plot(np.mean(thickness_all[3][0]), np.mean(thickness_all[3][2]), 'yo')
plt.plot([0.0002,0.0009], [0.0002,0.0009],color='black')
plt.show()
# %% Plot spacing
plt.figure()
plt.plot(spacing_all[0][0], spacing_all[0][2], 'b+')
plt.plot(spacing_all[1][0], spacing_all[1][2], 'g+')
plt.plot(spacing_all[2][0], spacing_all[2][2], 'r+')
plt.plot(spacing_all[3][0], spacing_all[3][2], 'y+')
plt.plot(np.mean(spacing_all[0][0]), np.mean(spacing_all[0][2]), 'bo')
plt.plot(np.mean(spacing_all[1][0]), np.mean(spacing_all[1][2]), 'go')
plt.plot(np.mean(spacing_all[2][0]), np.mean(spacing_all[2][2]), 'ro')
plt.plot(np.mean(spacing_all[3][0]), np.mean(spacing_all[3][2]), 'yo')
plt.plot([0.0002,0.0005], [0.0002,0.0005],color='black')
plt.show()

# %%
