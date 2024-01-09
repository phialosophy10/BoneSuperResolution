#!/bin/sh
#BSUB -q gpua100
#BSUB -J srgan
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu40gb]"
#BSUB -o /work3/soeba/HALOS/experiments/srgan/train_%J.out
#BSUB -e /work3/soeba/HALOS/experiments/srgan/train_%J.err

cd ..

source init.sh

python -u srgan_bones.py
