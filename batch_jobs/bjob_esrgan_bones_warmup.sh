#!/bin/sh
#BSUB -q gpua100
#BSUB -J esrgan
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -o /work3/soeba/HALOS/experiments/esrgan/train_%J.out
#BSUB -e /work3/soeba/HALOS/experiments/esrgan/train_%J.err

cd ..

source init.sh

sleep 15

python -u esrgan_bones_warmup.py
