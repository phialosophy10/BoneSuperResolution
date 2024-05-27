# CT Superresolution of human bones

This is the project page for the paper titled "Superresolution of real-world multiscale bone CT verified with clinical bone measures", presented at MIUA conference 2024 in Manchester.

The project explores the performance of superresolution GANs on two real-world multiscale datasets and compares with the performance on synthetically downscaled data. Evaluation is carried out with both image similarity metrics PSNR and SSIM, as well as with clinically relevant bone measures, calculated on the resulting superresolution bone images.

## Data description and download

The FACTS (Femur Archaeological CT Superresolution) dataset consists of 13 archaeological proximal femurs from humans dated around the Middle Ages in Denmark. The bones have been scanned using a SIEMENS clinical CT scanner resulting in (0.21x0.21x0.4 mm³) resolution, as well as using a NIKON micro-CT scanner resulting in (58x58x58 um³) resolution. The volumes have been registered and the clinical volume resliced to the same size, giving voxel-to-voxel correspondance.

The data can be downloaded [here](https://github.com/phialosophy10/BoneSuperResolution) (CORRECT LINK TO DATA WILL BE PROVIDED SOON)

## Usage

```python
import tifffile
import matplotlib.pyplot as plt

# load femur "001"
vol_hr = tifffile.imread('/root_path/f_001/HR/f_001.tiff')
vol_lr = tifffile.imread('/root_path/f_001/LR/f_001.tiff')
vol_sy = tifffile.imread('/root_path/f_001/SY/f_001.tiff')

# show selected slice from each volume
idx = 800

fig, ax = plt.subplots(1, 3)
ax[0] = plt.imshow(vol_hr[:,:,idx],cmap='gray')
ax[0].set_title('micro-CT slice')
ax[1] = plt.imshow(vol_hr[:,:,idx],cmap='gray')
ax[0].set_title('clinical CT slice')
ax[2] = plt.imshow(vol_hr[:,:,idx],cmap='gray')
ax[0].set_title('synthetic low-res slice')
plt.show()


```
