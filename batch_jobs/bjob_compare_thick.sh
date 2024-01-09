#!/bin/sh
#BSUB -q gpua100
#BSUB -J comp_thick
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -o /work3/soeba/HALOS/experiments/comp_thick/%J.out
#BSUB -e /work3/soeba/HALOS/experiments/comp_thick/%J.err

cd ..

source init.sh

python -u compare_thickness_all.py
