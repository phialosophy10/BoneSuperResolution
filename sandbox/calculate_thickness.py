# %% Packages
import nibabel as nib
import numpy as np
import porespy as ps
import scipy.ndimage as spim
import matplotlib.pyplot as plt
ps.visualization.set_mpl_style()

# %% Load a slice
#vol_m = nib.load('/work3/soeba/HALOS/Data/microCT/femur74_mod.nii')
vol_m = nib.load('/work3/soeba/HALOS/Data/clinicalCT/resliced_clinical_femur74.nii.gz')
im_m = np.array(vol_m.dataobj[:,:,400])
#im_m = im_m[250:450,550:750]

# im = np.zeros([300, 300])
# im = ps.generators.rsa(im, r=20, volume_fraction=0.2)
# im = ps.generators.rsa(im, r=15, volume_fraction=0.4)
# im = ps.generators.rsa(im, r=10, volume_fraction=0.6)
# im_m = im == 0
# fig, ax = plt.subplots()
# ax.imshow(im, interpolation='none', origin='lower')
# ax.axis(False)

# %% Make binary
im_m[im_m<=65] = 0
im_m[im_m>65] = 1

# %% Show slice
plt.imshow(im_m)
plt.colorbar()
plt.show()

# %% Local Thickness
thk = ps.filters.local_thickness(im_m, mode='dt')
fig, ax = plt.subplots()
ax.imshow(thk, interpolation='none', origin='lower', cmap=plt.cm.jet)
ax.axis(False)
# %%
