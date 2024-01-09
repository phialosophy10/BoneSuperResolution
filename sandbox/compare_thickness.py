# %% Packages
import nibabel as nib
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
ps.visualization.set_mpl_style()

# %% Load a slice
slice_no = 800
vol_m = nib.load('/work3/soeba/HALOS/Data/microCT/femur01_mod.nii')
vol_c = nib.load('/work3/soeba/HALOS/Data/clinicalCT/resliced_femur01.nii.gz')
im_m = np.array(vol_m.dataobj[:,:,slice_no])
im_c = np.array(vol_c.dataobj[:,:,slice_no])

# %% Make binary
im_m[im_m<=65] = 0
im_m[im_m>65] = 1

im_c[im_c<=65] = 0
im_c[im_c>65] = 1

# %% Local Thicknesses
thk_m = ps.filters.local_thickness(im_m, mode='dt')
thk_c = ps.filters.local_thickness(im_c, mode='dt')

# %% Show slices and thickness
fig, axs = plt.subplots(2, 2, sharey=True, figsize=(9,7))
axs[0,0].imshow(im_c)
axs[0,1].imshow(im_m)
axs[1,0].imshow(thk_c, interpolation='none', origin='upper', cmap=plt.cm.jet)
axs[1,1].imshow(thk_m, interpolation='none', origin='upper', cmap=plt.cm.jet)
for ax in fig.get_axes():
    ax.label_outer()
plt.suptitle(f'Slice no: %s' % (slice_no))
plt.show()

# %%
