# %% Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# %% Load a volume

path = '/work3/soeba/HALOS/Data/'

vol_m = nib.load(path+'microCT/femur01_mod.nii')
vol_c = nib.load(path+'clinicalCT/resliced_clinical_femur01.nii.gz')

vol_m = np.array(vol_m.dataobj)
#vol_m = vol_m.transpose((1, 0, 2)) # For femur15
vol_c = np.array(vol_c.dataobj)
#vol_c = vol_c.transpose((1, 0, 2)) # For femur15

# %% Select ROI

vol_m = vol_m[3:,102:614,200:1175]
vol_c = vol_c[3:,102:614,200:1175]

# vol_m = vol_m[:,:,150:1200]
# vol_c = vol_c[:,:,150:1200]

# vol_m = vol_m[:,:,125:1150]
# vol_c = vol_c[:,:,125:1150]

# vol_m = vol_m[:,:,150:1250]
# vol_c = vol_c[:,:,150:1250]

# %% Make binary
# vol_m[vol_m<=65] = 0
# vol_m[vol_m>65] = 1

# vol_c[vol_c<=65] = 0
# vol_c[vol_c>65] = 1

# %% Save volume as images slices

# np.save(path+'Images/m03',vol_m)
# np.save(path+'Images/c03',vol_c)

# for i in range(vol_m.shape[2]):
#     np.save(path+'Images/test/hr/m00_'+str(i).zfill(4),vol_m[:,:,i])
#     np.save(path+'Images/test/lr/c00_'+str(i).zfill(4),vol_c[:,:,i])

n_ims = vol_m.shape[2]
split_ind = int(n_ims/5)
inds = list(range(n_ims))
random.shuffle(inds)
test_inds = inds[:split_ind]
train_inds = inds[split_ind:]

for i in range(len(test_inds)):
    im_m = Image.fromarray(vol_m[:,:,test_inds[i]]).convert("L")
    im_m.save(path+'Images/test/hr/m00_'+str(test_inds[i]).zfill(4)+'.jpg')
    im_c = Image.fromarray(vol_c[:,:,test_inds[i]]).convert("L")
    im_c.save(path+'Images/test/lr/c00_'+str(test_inds[i]).zfill(4)+'.jpg')

for i in range(len(train_inds)):
    im_m = Image.fromarray(vol_m[:,:,train_inds[i]]).convert("L")
    im_m.save(path+'Images/train/hr/m00_'+str(train_inds[i]).zfill(4)+'.jpg')
    im_c = Image.fromarray(vol_c[:,:,train_inds[i]]).convert("L")
    im_c.save(path+'Images/train/lr/c00_'+str(train_inds[i]).zfill(4)+'.jpg')

# %%
