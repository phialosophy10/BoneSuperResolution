# %% Packages
import localthickness as lt
import numpy as np
import matplotlib.pyplot as plt
import glob
import porespy as ps

# %% Set data path
hr_path = "/work3/soeba/HALOS/Data/Images/micro/01a_0800_08.npy"
lr_path = "/work3/soeba/HALOS/Data/Images/clinical/low-res/linear/01a_0800_08.npy"
sr_path = "/work3/soeba/HALOS/Data/Images/SR/low-res/linear/01a_0800_08.npy"
mask_path = "/work3/soeba/HALOS/Data/Images/masks/01a_0800_08.npy"

# %% Show images
plt.figure()
plt.imshow(np.load(hr_path),cmap='gray')
plt.show()

plt.figure()
plt.imshow(np.load(lr_path),cmap='gray')
plt.show()

plt.figure()
plt.imshow(np.load(sr_path),cmap='gray')
plt.show()

plt.figure()
plt.imshow(np.load(mask_path),cmap='gray')
plt.show()

# %% Load images, make binary and inverse binary
im_m = np.load(hr_path) > 75
im_c = np.load(lr_path) > 60
im_s = np.uint8(np.load(sr_path) * 255) > 75

# Show binary images
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
axs[0].imshow(im_m)
axs[1].imshow(im_c)
axs[2].imshow(im_s)
plt.suptitle('Binary images (thickness)')
plt.show()

# Make inverse binary images (for spacing calculations)
mask = np.load(mask_path)
im_m_sp = ~im_m
im_c_sp = ~im_c
im_s_sp = ~im_s
im_m_sp[mask==0] = False
im_s_sp[mask==0] = False

# Show inverse binary images
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
axs[0].imshow(im_m_sp) #im_m_sp #~im_m
axs[1].imshow(im_c_sp) #im_c_sp #~im_c
axs[2].imshow(im_s_sp) #im_s_sp #~im_s
plt.suptitle('Binary images (spacing)')
plt.show()

# %% Compute thickness and separation
thk_m = lt.local_thickness(im_m, scale=0.5)
thk_c = lt.local_thickness(im_c, scale=0.5)
thk_s = lt.local_thickness(im_s, scale=0.5)

sp_m = lt.local_thickness(im_m_sp, scale=0.5) #im_m_sp
sp_c = lt.local_thickness(im_c_sp, scale=0.5) #im_c_sp
sp_s = lt.local_thickness(im_s_sp, scale=0.5) #im_s_sp

# Show thickness images
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
axs[0].imshow(thk_m)
axs[1].imshow(thk_c)
axs[2].imshow(thk_s)
plt.suptitle('Thickness images')
plt.show()

# Show spacing images
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9,4))
axs[0].imshow(sp_m)
axs[1].imshow(sp_c)
axs[2].imshow(sp_s)
plt.suptitle('Spacing images')
plt.show()

# %% Calculate poresize distribution for thickness
pore_sizes = list(np.linspace(1.0, 50, num=25))
voxel_size= 0.12 * 1e-3

data_m = ps.metrics.pore_size_distribution(im=thk_m, bins=pore_sizes, log=False)
data_c = ps.metrics.pore_size_distribution(im=thk_c, bins=pore_sizes, log=False)
data_s = ps.metrics.pore_size_distribution(im=thk_s, bins=pore_sizes, log=False)

# Plot histograms
fig, ax = plt.subplots(1, 3, figsize=[15, 4])
ax[0].bar(data_m.bin_centers*voxel_size, data_m.pdf, data_m.bin_widths*voxel_size, edgecolor='k')
ax[1].bar(data_c.bin_centers*voxel_size, data_c.pdf, data_c.bin_widths*voxel_size, edgecolor='k')
ax[2].bar(data_s.bin_centers*voxel_size, data_s.pdf, data_s.bin_widths*voxel_size, edgecolor='k')
ax[0].set_title('Thickness, HR')
ax[1].set_title('Thickness, LR')
ax[2].set_title('Thickness, SR')
plt.suptitle('Thickness histograms')

# Calculate poresize distribution for spacing
data_m_sp = ps.metrics.pore_size_distribution(im=sp_m, bins=pore_sizes, log=False)
data_c_sp = ps.metrics.pore_size_distribution(im=sp_c, bins=pore_sizes, log=False)
data_s_sp = ps.metrics.pore_size_distribution(im=sp_s, bins=pore_sizes, log=False)

# Plot histograms
fig, ax = plt.subplots(1, 3, figsize=[15, 4])
ax[0].bar(data_m_sp.bin_centers*voxel_size, data_m_sp.pdf, data_m_sp.bin_widths*voxel_size, edgecolor='k')
ax[1].bar(data_c_sp.bin_centers*voxel_size, data_c_sp.pdf, data_c_sp.bin_widths*voxel_size, edgecolor='k')
ax[2].bar(data_s_sp.bin_centers*voxel_size, data_s_sp.pdf, data_s_sp.bin_widths*voxel_size, edgecolor='k')
ax[0].set_title('Spacing, HR')
ax[1].set_title('Spacing, LR')
ax[2].set_title('Spacing, SR')
plt.suptitle('Spacing histograms')
# %%
