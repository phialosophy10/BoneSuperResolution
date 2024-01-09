#!/bin/sh
#BSUB -q gpuv100
#BSUB -J nii2dicom
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/nii2dicom/%J.out
#BSUB -e /work3/soeba/HALOS/experiments/nii2dicom/%J.err

cd ..

source init.sh

python -u nifti2dicom.py


