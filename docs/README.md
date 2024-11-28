# FACTS (Femur Archaeological CT Superresolution) Dataset

This is the project page for the FACTS dataset, published simultaneously with the paper ["Superresolution of real-world multiscale bone CT verified with clinical bone measures"](https://doi.org/10.1007/978-3-031-66958-3_12), presented at the MIUA (Medical Image Understanding and Analysis) conference 2024 in Manchester.

The dataset was created as a benchmark for 3D superresolution and consists of archaeological bones scanned with two different scanners, resulting in a multiscale dataset, that is not simply produced by downsampling high-resolution volumes.

A 3D rendering of one of the full micro-CT bone scans, with the femoral neck highlighted, can be seen in the figure below:

![3D rendering of femur](https://github.com/phialosophy10/BoneSuperResolution/assets/93533251/945d4ad4-9023-4e59-9b42-a36c5e1b2978)

## Citing
If you use the dataset in your research, you should cite the [paper](https://doi.org/10.1007/978-3-031-66958-3_12), where it was first published (DOI: 10.1007/978-3-031-66958-3_12). You can download the Bibtex file from the "Cite this repository" drop-down menu in the [repo](https://github.com/phialosophy10/BoneSuperResolution).

## Download

The data can be downloaded [here](https://archive.compute.dtu.dk/files/public/projects/FACTS).

### Download using 'wget'
To do a stable download, you can follow these steps: 
* Copy the download link from the online archive (e.g. "https://archive.compute.dtu.dk/files/public/projects/FACTS/f_001.zip")
* Open a terminal on your workspace and navigate to the folder, where you want the data
* Use the following command:

````
wget -c -O my_data.tif "paste the url within quotes"
````
e.g.

````
wget -c -O f_001.tif "https://archive.compute.dtu.dk/files/public/projects/FACTS/f_001.zip"
````

## Detailed description
The FACTS dataset consists of 13 archaeological proximal femurs from humans dated around the Middle Ages in Denmark. The dataset contains both left and right proximal femurs from both males and females (2M/11F). 

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

### Scanning modalities
The bones have been scanned using a SIEMENS clinical CT scanner resulting in a (0.21x0.21x0.4 mm³) resolution, named the low-resolution (LR) scan, as well as with a NIKON micro-CT scanner resulting in (58x58x58 um³) resolution, named the high-resolution (HR) scan. The HR and LR volumes for each of the 13 bones have been registered using ITK-SNAP and the LR volume has then been resliced to the same voxelsize (using linear interpolation), giving voxel-to-voxel correspondance.

For research purposes, we have also produced synthetic low-resolution volumes (SY), created by 4x downsampling using linear interpolation and filtering with a Gaussian kernel w. $\sigma = 1.2$. This processing was chosen to match what is typically done on datasets produced for testing superresolution architectures.

Lastly, we have produced binary mask volumes (MS) of the bones by performing blurring, dilation and thresholding of the HR volumes. The masks indicate whether a voxel is inside (255) or outside (0) of the bone surface.

An overview of the modalities along with file types, data types and some voxel statistics are given in the table below:

|     Modality              |     Orig. resolution        |     File type        |     Data type        |     Voxel intensity stats.     (avg.)                |
|---------------------------|-----------------------------|----------------------|----------------------|------------------------------------------------------|
|     Micro-CT (HR)         |     58x58x58µm3             |     TIFF             |     Float32          | [Min/Max]: [-100/1000],  [Mean/Median]: [15.3/1.4]   |
|     Clinical CT   (LR)    |     .21x.21x.4mm3           |     TIFF             |     Float32          | [Min/Max]: [-1000/3000], [Mean/Median]: [-736/-982]  |
|     Synthetic   (SY)      | -                           |     TIFF             |     Float32          | -                                                    |
|     Mask (MS)             | -                           |     TIFF             |     Uint8            | -                                                    |

### Visualizations
The goal of superresolution for bone data is typically to recover microstructure, which is very detailed in the trabecular bone (inner part) in particular. Due to limits on scan time and radiation doses, clinical scans are typically not of high enough resolution to show clear microstructure. To get a sense of the structure in the scans, example slices with zoomed-in patches of the clinical CT, micro-CT and synthetically downsampled data can be seen in the figure below:

![Microstructure of data](https://github.com/user-attachments/assets/4a5e778e-2bc9-4299-880a-32f57b2e8280)

We have created a GIF that shows the centre slice of a bone in all three planes for all modalities here:

![Bone GIF (1)](https://github.com/user-attachments/assets/d8f591f7-f5bc-4677-8526-9799362ccce4)

## Usage

```python
import tifffile
import matplotlib.pyplot as plt

# load femur "f_001"
vol_hr = tifffile.imread('/root_path/f_001/HR/f_001.tif')
vol_lr = tifffile.imread('/root_path/f_001/LR/f_001.tif')
vol_sy = tifffile.imread('/root_path/f_001/SY/f_001.tif')
vol_ms = tifffile.imread('/root_path/f_001/MS/f_001.tif')

# show selected slice from each volume
idx = 800

fig, ax = plt.subplots(1, 4)
ax[0] = plt.imshow(vol_hr[:,:,idx],cmap='gray')
ax[0].set_title('micro-CT slice')
ax[1] = plt.imshow(vol_lr[:,:,idx],cmap='gray')
ax[1].set_title('clinical CT slice')
ax[2] = plt.imshow(vol_sy[:,:,idx],cmap='gray')
ax[2].set_title('synthetic low-res slice')
ax[3] = plt.imshow(vol_ms[:,:,idx],cmap='gray')
ax[3].set_title('mask slice')
plt.show()


```

## Benchmarks on the data with GAN architectures

The work presented in the [paper](https://doi.org/10.1007/978-3-031-66958-3_12) performs superresolution on the FACTS dataset along with a private multiscale dataset provided by the University of California San Francisco (UCSF) and compares this with superresolution on the synthetically downscaled counterparts. The SRGAN[^1] and ESRGAN[^2] architectures are used as benchmarks and evaluation is carried out with both image similarity metrics PSNR and SSIM, as well as with clinically relevant bone measures, calculated on the resulting superresolution volumes and compared with the micro-CT ground truth volumes.

In the figure below is shown selected results on the FACTS dataset. The left image shows results for the ESRGAN model trained and tested on real data, the center image shows results for the ESRGAN model trained and tested on synthetic data, and the right image shows results for the SRGAN model trained and tested on real data.

![Selected SR results](https://github.com/phialosophy10/BoneSuperResolution/assets/93533251/73cf9ea8-0ace-49cd-8df8-d157a88b458f)


## References
[^1]: C. Ledig et. al, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 105-114. [DOI](doi.org/10.1109/CVPR.2017.19).

[^2]: X Wang et. al, ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks, Leal-Taixé, L., Roth, S. (eds) Computer Vision – ECCV 2018 Workshops. ECCV 2018. Lecture Notes in Computer Science(), vol 11133. Springer, Cham. [DOI](doi.org/10.1007/978-3-030-11021-5_5).

