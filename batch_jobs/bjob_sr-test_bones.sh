#!/bin/sh
#BSUB -q gpuv100
#BSUB -J sr-test
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/sr-test/train_%J.out
#BSUB -e /work3/soeba/HALOS/experiments/sr-test/train_%J.err

cd ..

source init.sh

sleep 15

python -u sr_test_bones_simple.py
