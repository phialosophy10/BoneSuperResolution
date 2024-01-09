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

python -u tiftonifti_ct.py 001
python -u tiftonifti_ct.py 002
python -u tiftonifti_ct.py 013
python -u tiftonifti_ct.py 015
python -u tiftonifti_ct.py 021
python -u tiftonifti_ct.py 026
python -u tiftonifti_ct.py 031
python -u tiftonifti_ct.py 074
python -u tiftonifti_ct.py 075
python -u tiftonifti_ct.py 083
python -u tiftonifti_ct.py 086
python -u tiftonifti_ct.py 138
python -u tiftonifti_ct.py 164
python -u tiftonifti_ct.py 172



