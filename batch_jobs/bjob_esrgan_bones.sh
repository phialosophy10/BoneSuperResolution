#!/bin/sh
#BSUB -q gpua100
#BSUB -J esrgan
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -o /work3/soeba/HALOS/experiments/esrgan/train_%J.out
#BSUB -e /work3/soeba/HALOS/experiments/esrgan/train_%J.err

cd ..

source init.sh

# args: 
# n_epochs (int)
# batch_size (int)
# learning rate (float)
# loss-coefficient (float)
# femur to test on (str, {"01", "15", "21", "74"})
# femur(s) to train on (str, {"01", "15", "21", "74"})

# e.g.: python -u esrgan_bones.py 10 8 0.00008 1 "01" "15"
# or: python -u esrgan_bones.py 20 8 0.00016 0.5 "15" "01" "21" "74"

python -u esrgan_bones.py 5 8 0.00016 1 "15" "01"
