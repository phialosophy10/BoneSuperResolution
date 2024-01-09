#!/bin/sh
#BSUB -q gpuv100
#BSUB -J tif_to_nifty
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/tifstack_to_nifty/%J.out
#BSUB -e /work3/soeba/HALOS/experiments/tifstack_to_nifty/%J.err

cd ..

source init.sh

python -u tiftonifti_mask.py 001
python -u tiftonifti_mask.py 002
python -u tiftonifti_mask.py 013
python -u tiftonifti_mask.py 015
python -u tiftonifti_mask.py 021
python -u tiftonifti_mask.py 026
python -u tiftonifti_mask.py 031
python -u tiftonifti_mask.py 074
python -u tiftonifti_mask.py 075
python -u tiftonifti_mask.py 083
python -u tiftonifti_mask.py 086
python -u tiftonifti_mask.py 138
python -u tiftonifti_mask.py 164
python -u tiftonifti_mask.py 172



