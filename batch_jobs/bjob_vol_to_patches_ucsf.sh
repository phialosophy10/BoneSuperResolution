#!/bin/sh
#BSUB -q gpuv100
#BSUB -J vol_to_slices
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/vol_to_slices/%J.out
#BSUB -e /work3/soeba/HALOS/experiments/vol_to_slices/%J.err

cd ..

source init.sh

python -u vol_to_slices_ucsf.py "05-01"
python -u vol_to_slices_ucsf.py "05-02"
python -u vol_to_slices_ucsf.py "05-03"
python -u vol_to_slices_ucsf.py "05-04"
python -u vol_to_slices_ucsf.py "05-05"


