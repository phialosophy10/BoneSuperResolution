#!/bin/sh
#BSUB -q gpua100
#BSUB -J srresnet
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -o /work3/soeba/HALOS/experiments/srresnet/train_%J.out
#BSUB -e /work3/soeba/HALOS/experiments/srresnet/train_%J.err

cd ..

source init.sh

sleep 15

python -u srresnet_bones.py
