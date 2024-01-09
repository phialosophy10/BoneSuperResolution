#!/bin/sh
#BSUB -q gpuv100
#BSUB -J unet
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 05:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/unet/train_%J.out
#BSUB -e /work3/soeba/HALOS/experiments/unet/train_%J.err

cd ..

source init.sh

python -u unet_bones.py
