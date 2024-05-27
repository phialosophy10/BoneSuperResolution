# CT Superresolution of human bones

This is the project page for the paper titled "Superresolution of real-world multiscale bone CT verified with clinical bone measures", presented at MIUA conference 2024 in Manchester.

The project explores the performance of superresolution GANs on two real-world multiscale datasets and compares with the performance on synthetically downscaled data. Evaluation is carried out with both image similarity metrics PSNR and SSIM, as well as with clinically relevant bone measures, calculated on the resulting superresolution bone images.

## Data description and download

The FACTS (Femur Archaeological CT Superresolution) dataset consists of 13 archaeological proximal femurs from humans dated around the Middle Ages in Denmark. The bones have been scanned using a SIEMENS clinical CT scanner resulting in (0.21x0.21x0.4 mm³) resolution, as well as using a NIKON micro-CT scanner resulting in (58x58x58 um³) resolution. The volumes have been registered and the clinical volume resliced to the same size, giving voxel-to-voxel correspondance.

The data can be downloaded [here](https://github.com/phialosophy10/BoneSuperResolution) (CORRECT LINK TO DATA WILL BE PROVIDED SOON)

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
