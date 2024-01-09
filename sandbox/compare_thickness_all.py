# %% Packages
import numpy as np
import localthickness as lt
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

# %% Calculate EMD for HR and SR
# def EMD(p, q):
#     x = p - q
#     y = cumsum(x, axis=0)
#     return abs(y).sum()

# EMD_thk = EMD(data_m.pdf, data_s.pdf)
# EMD_sp = EMD(data_m_sp.pdf, data_s_sp.pdf)

# print(f'Earth Movers Distance between HR and SR (Thickness): {EMD_thk}\n')
# print(f'Earth Movers Distance between HR and SR (Spacing): {EMD_sp}\n')

# %% Calculate thickness and spacing
voxel_size= 0.12 * 1e-3 #voxel size is 120 micrometers (?)
bone_thresh = 75
bone_thresh_lr = 40
n_patches = len(hr_paths[0]) // 8 #20
start_slice = -500

pore_sizes = list(np.linspace(1.0, 50, num=25))

thickness_all, spacing_all = [], []
for j in range(len(femur_no)):
    thick_m, thick_s, space_m, space_s = [], [], [], []
    # thick_c, space_c = [], []
    for i in range(n_patches): #n_patches #len(hr_paths)
        im_m = np.load(hr_paths[j][start_slice-i]) > bone_thresh
        # im_c = np.load(lr_paths[j][start_slice-i]) > bone_thresh_lr
        im_s = np.uint8(np.load(sr_paths[j][start_slice-i]) * 255) > bone_thresh

        mask = np.load(mask_paths[j][start_slice-i])
        im_m_sp = ~im_m
        # im_c_sp = ~im_c
        im_s_sp = ~im_s
        im_m_sp[mask==0] = False
        im_s_sp[mask==0] = False

        # Calculate local thickness images
        thk_m = lt.local_thickness(im_m, scale=0.5)
        # thk_c = lt.local_thickness(im_c, scale=0.5)
        thk_s = lt.local_thickness(im_s, scale=0.5)

        # Calculate local spacing images
        sp_m = lt.local_thickness(im_m_sp, scale=0.5)
        # sp_c = lt.local_thickness(im_c_sp, scale=0.5)
        sp_s = lt.local_thickness(im_s_sp, scale=0.5)

        # Calculate poresize distribution for thickness
        data_m = ps.metrics.pore_size_distribution(im=thk_m, bins=pore_sizes, log=False)
        # data_c = ps.metrics.pore_size_distribution(im=thk_c, bins=pore_sizes, log=False)
        data_s = ps.metrics.pore_size_distribution(im=thk_s, bins=pore_sizes, log=False)

        # Add to thickness
        if not np.isnan(np.sum(data_m.pdf)):
            thick_m.append(np.sum(data_m.bin_centers*voxel_size*data_m.pdf)/np.sum(data_m.pdf))
        # if not np.isnan(np.sum(data_c.pdf)):
        #     thick_c.append(np.sum(data_c.bin_centers*voxel_size*data_c.pdf)/np.sum(data_c.pdf))
        if not np.isnan(np.sum(data_s.pdf)):
            thick_s.append(np.sum(data_s.bin_centers*voxel_size*data_s.pdf)/np.sum(data_s.pdf))

        # Calculate poresize distribution for thickness
        data_m_sp = ps.metrics.pore_size_distribution(im=sp_m, bins=pore_sizes, log=False)
        # data_c_sp = ps.metrics.pore_size_distribution(im=sp_c, bins=pore_sizes, log=False)
        data_s_sp = ps.metrics.pore_size_distribution(im=sp_s, bins=pore_sizes, log=False)

        # Add to spacing
        if not np.isnan(np.sum(data_m_sp.pdf)):
            space_m.append(np.sum(data_m_sp.bin_centers*voxel_size*data_m_sp.pdf)/np.sum(data_m_sp.pdf))
        # if not np.isnan(np.sum(data_c_sp.pdf)):
        #     space_c.append(np.sum(data_c_sp.bin_centers*voxel_size*data_c_sp.pdf)/np.sum(data_c_sp.pdf))
        if not np.isnan(np.sum(data_s_sp.pdf)):
            space_s.append(np.sum(data_s_sp.bin_centers*voxel_size*data_s_sp.pdf)/np.sum(data_s_sp.pdf))
    thickness_all.append([thick_m, thick_s]) #thick_c, 
    spacing_all.append([space_m, space_s]) #space_c, 

# %% Plot thickness
plt.figure(figsize=(15,10))
plt.plot(thickness_all[0][0], thickness_all[0][1], 'b+', label='Femur 01')
plt.plot(thickness_all[1][0], thickness_all[1][1], 'g+', label='Femur 15')
plt.plot(thickness_all[2][0], thickness_all[2][1], 'r+', label='Femur 21')
plt.plot(thickness_all[3][0], thickness_all[3][1], 'y+', label='Femur 74')
plt.plot(np.mean(thickness_all[0][0]), np.mean(thickness_all[0][1]), 'bo')
plt.plot(np.mean(thickness_all[1][0]), np.mean(thickness_all[1][1]), 'go')
plt.plot(np.mean(thickness_all[2][0]), np.mean(thickness_all[2][1]), 'ro')
plt.plot(np.mean(thickness_all[3][0]), np.mean(thickness_all[3][1]), 'yo')
plt.plot([0.0002,0.0011], [0.0002,0.0011],color='black')
plt.legend()
plt.title('Local thickness')
plt.savefig('/work3/soeba/HALOS/experiments/comp_thick/plots/thickness')
plt.show()

# %% Plot spacing
plt.figure(figsize=(15,10))
plt.plot(spacing_all[0][0], spacing_all[0][1], 'b+', label='Femur 01')
plt.plot(spacing_all[1][0], spacing_all[1][1], 'g+', label='Femur 15')
plt.plot(spacing_all[2][0], spacing_all[2][1], 'r+', label='Femur 21')
plt.plot(spacing_all[3][0], spacing_all[3][1], 'y+', label='Femur 74')
plt.plot(np.mean(spacing_all[0][0]), np.mean(spacing_all[0][1]), 'bo')
plt.plot(np.mean(spacing_all[1][0]), np.mean(spacing_all[1][1]), 'go')
plt.plot(np.mean(spacing_all[2][0]), np.mean(spacing_all[2][1]), 'ro')
plt.plot(np.mean(spacing_all[3][0]), np.mean(spacing_all[3][1]), 'yo')
plt.plot([0.0002,0.0005], [0.0002,0.0005],color='black')
plt.legend()
plt.title('Local spacing')
plt.savefig('/work3/soeba/HALOS/experiments/comp_thick/plots/spacing')
plt.show()

# %%
