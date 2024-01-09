## This script loads a tif-file containng a 3D volume and saves it as a NiFTI-file.
## You need to input the origin and voxel-spacings if these are known.

# Packages
from skimage import io
import numpy as np
import nibabel as nib

# Load .tif stack (if you have volume in one stacked file)
path = "/path/to/file/"
vol = io.imread(path + "file_name.tif")

# Transpose volume array, so that it becomes (x, y, z)
vol = vol.transpose((1, 2, 0))

# Simple header for volume
origin = np.array([0,0,0])  # Location of corner voxel, use [0,0,0] if nothing else is known from the scanner

affine = np.zeros((4,4))
affine[0:3,3] = origin
affine[0,0] = 0.0573016     # spacing in x [mm]
affine[1,1] = 0.0573016     # spacing in y [mm]
affine[2,2] = 0.0573016     # spacing in z [mm]

# Save volume as NiFTI
niiVol = nib.Nifti1Image(vol, affine)
nib.save(niiVol, path + "new_file_name.nii")