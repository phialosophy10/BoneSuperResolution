# %% Packages
import numpy as np
import nibabel as nib
import scipy.ndimage
import matplotlib.pyplot as plt

#%% MICRO CT
# Load 
volMu = nib.load('/work3/soeba/HALOS/Data/microCT/femur15.nii')

# Transform origin to clinical geometrical center
tform_mu = volMu.affine

# Recast to uint8
intLimit = [200, 6000] # femur15: [200, 6000], femur01/21/74: [2000, 30000]

vol = np.array(volMu.dataobj) #The actual data matrix copied
vol = np.float32(vol)
vol[vol < intLimit[0]] = intLimit[0]
vol[vol > intLimit[1]] = intLimit[1]
vol = vol - intLimit[0] # intensity range 0 to (max-min)
vol = vol * (1/np.max(vol)) # intensity range 0 to 1
vol = np.uint8(vol*255)

# %% Histogram
plt.hist(vol.ravel(), bins=40, density=True)
plt.show()

# %% Scale to a factor of 0.5
factor = 0.5
vol = scipy.ndimage.zoom(vol, factor, order=1)

#%%
# Reset origin
tform_mu[0:3,3] = [0, 96, -815]
tform_mu[0,0] = tform_mu[0,0] * 1/factor
tform_mu[1,1] = tform_mu[1,1] * 1/factor
tform_mu[2,2] = tform_mu[2,2] * 1/factor

niiVol = nib.Nifti1Image(vol, tform_mu)
nib.save(niiVol, '/work3/soeba/HALOS/Data/microCT/femur15_mod.nii')  
# %%
