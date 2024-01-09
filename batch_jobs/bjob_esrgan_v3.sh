#!/bin/sh
#BSUB -q gpua100
#BSUB -J esrgan
#BSUB -n 8
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
# loss-coefficient, adversarial (float)
# loss-coefficient, pixel (float)
# loss-coefficient, content (float)
# number of warmup-batches with only pixel-wise loss (int)
# femur to test on (str)
# femur(s) to train on (str)
# femurs: {"001","002","013","015","021","026","074","075","083","086","138","164","172"}

python -u esrgan_bones_v3-5.py 1 4 1e-4 1e-3 1 6e-3 0 "001" "002" "026" "083"
