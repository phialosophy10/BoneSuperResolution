# FACTS (Femur Archaeological CT Superresolution) Dataset

This is the project page for the FACTS dataset, published simultaneously with the paper "Superresolution of real-world multiscale bone CT verified with clinical bone measures", presented at the MIUA (Medical Image Understanding and Analysis) conference 2024 in Manchester.
DOI: [10.1007/978-3-031-66958-3_12](https://doi.org/10.1007/978-3-031-66958-3_12)

The dataset was created as a benchmark for 3D superresolution networks and consists of archaeological bones scanned with two different scanners, resulting in a multiscale dataset, that is not simply produced by downsampling high-resolution volumes.

## Detailed description and download

The data can be downloaded [here](https://github.com/phialosophy10/BoneSuperResolution) (CORRECT LINK TO DATA WILL BE PROVIDED SOON)

The FACTS (Femur Archaeological CT Superresolution) dataset consists of 13 archaeological proximal femurs from humans dated around the Middle Ages in Denmark. The dataset contains both left and right proximal femurs from both males and females (2M/11F). 

An overview of the names, sex, approximated age (if available) and the shape of each of the bone volumes is given in the following table:

| Name     | Sex    | Age                | Shape                 |
|----------|--------|--------------------|-----------------------|
| f_001    | F      | Middle-aged        | [904, 1348, 1561]     |
| f_002    | F      | unknown            | [824, 1492, 1554]     |
| f_013    | F      | unknown            | [976, 1202, 1501]     |
| f_015    | M      | Middle-aged/old    | [1474, 1198, 1798]    |
| f_021    | M      | Young              | [1198, 1498, 1697]    |
| f_026    | F      | unknown            | [810, 1478, 1544]     |
| f_074    | F      | Young              | [982, 1618, 1595]     |
| f_075    | F      | unknown            | [846, 1424, 1600]     |
| f_083    | F      | unknown            | [1048, 1344, 1430]    |
| f_086    | F      | unknown            | [716, 1434, 1431]     |
| f_138    | F      | unknown            | [1228, 1226, 1412]    |
| f_164    | F      | unknown            | [914, 1266, 1367]     |
| f_172    | F      | unknown            | [1056, 1410, 1688]    |

The bones have been scanned using a SIEMENS clinical CT scanner resulting in a (0.21x0.21x0.4 mm³) resolution, named the low-resolution (LR) scan, as well as with a NIKON micro-CT scanner resulting in (58x58x58 um³) resolution, named the high-resolution (HR) scan. The HR and LR volumes for each of the 13 bones have been registered using ITK-SNAP and the LR volume has then been resliced to the same voxelsize (using linear interpolation), giving voxel-to-voxel correspondance.

For research purposes, we have also produced synthetic low-resolution volumes (SY), created by 4x downsampling using linear interpolation and filtering with a Gaussian kernel w. $\sigma = 1.2$. This processing was chosen to match what is typically done on datasets produced for testing superresolution architectures.

Lastly, we have produced binary mask volumes (MS) of the bones by performing blurring, dilation and thresholding of the HR volumes. The masks indicate whether a voxel is inside (255) or outside (0) of the bone surface.

On overview of the modalities along with file types, data types and some voxel statistics are given in the table below:

| **    Modality   **       | **    Orig. resolution   ** | **    File type   ** | **    Data type   ** | **    Voxel intensity stats.     (avg.)   ** |
|---------------------------|-----------------------------|----------------------|----------------------|----------------------------------------------|
|     Micro-CT (HR)         |     58x58x58µm3             |     TIFF             |     Float32          |     Min: -100                                |
|                           |                             |                      |                      |     Max: 1000                                |
|                           |                             |                      |                      |     Mean: 15.3                               |
|                           |                             |                      |                      |     Median: 1.4                              |
|     Clinical CT   (LR)    |     .21x.21x.4mm3           |                      |                      |     Min: -1000                               |
|                           |                             |                      |                      |     Max: 3000                                |
|                           |                             |                      |                      |     Mean: -736                               |
|                           |                             |                      |                      |     Median: -982                             |
|     Synthetic   (SY)      |                             |                      |                      |                                              |
|     Mask (MS)             |                             |                      |     Uint8            |                                              |

A 3D rendering of one of the full micro-CT bone scans, with the femoral neck highlighted, can be seen in the figure below:

![3D rendering of femur](https://github.com/phialosophy10/BoneSuperResolution/assets/93533251/945d4ad4-9023-4e59-9b42-a36c5e1b2978)

The goal of superresolution for bone data is typically to recover microstructure To get a sense of the structure in the scans, example slices with zoomed-in patches of the clinical CT, micro-CT and synthetically downsampled data can be seen in the figure below:

![Microstructure of data](https://github.com/user-attachments/assets/4a5e778e-2bc9-4299-880a-32f57b2e8280)

We also show the centre slices of the micro-CT, clinical CT, synthetic and mask of one of the bone scans in the figure below:

![Center slices of HR, LR, mask and synthetic](https://github.com/user-attachments/assets/fb850a91-5d7c-40c2-9224-57ff169c1eaf)

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

## Benchmarks on the data with GAN architectures

The work presented in the paper compares the performance of superresolution GANs on real-world multiscale datasets and synthetically downscaled datasets. Evaluation is carried out with both image similarity metrics PSNR and SSIM, as well as with clinically relevant bone measures, calculated on the resulting superresolution volumes and compared with the micro-CT ground truth volumes.

In the figure below is shown selected results on the FACTS dataset. The left image shows results for the ESRGAN model trained and tested on real data, the center image shows results for the ESRGAN model trained and tested on synthetic data, and the right image shows results for the SRGAN model trained and tested on real data.

![Selected SR results](https://github.com/phialosophy10/BoneSuperResolution/assets/93533251/73cf9ea8-0ace-49cd-8df8-d157a88b458f)


