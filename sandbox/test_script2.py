# %% Packages
import numpy as np
import matplotlib.pyplot as plt
import glob

from datetime import datetime, date, time

# %% Set paths
path = "/work3/soeba/HALOS/Data/"

m_paths = sorted(glob.glob(path + "Images/micro/*.*"))
thick_paths = sorted(glob.glob(path + "Images/thick/*.*"))
# c_h_paths = sorted(glob.glob(path + "Images/clinical/hi-res/*.*"))
# c_l_paths = sorted(glob.glob(path + "Images/clinical/low-res/linear/*.*"))
# c_n_paths = sorted(glob.glob(path + "Images/clinical/low-res/nn/*.*"))

# %% Load images
show_int = 40548
im_m = np.load(m_paths[show_int])
im_thick = np.load(thick_paths[show_int])
# im_c_h = np.load(c_h_paths[show_int])
# im_c_l = np.load(c_l_paths[show_int])
# im_c_n = np.load(c_n_paths[show_int])

# %% Plot images

plt.imshow(im_thick, cmap=plt.cm.jet)
plt.show()
# %%
im_select = np.zeros(im_thick.shape)
im_select[im_thick == np.unique(im_thick)[15]] = 1

plt.imshow(im_select)
plt.show()
# %%
