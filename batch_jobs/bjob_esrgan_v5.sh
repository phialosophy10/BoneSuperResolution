#!/bin/sh
#BSUB -q gpuv100
#BSUB -J esrgan
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[gpu32gb]"
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
# femur(s) to train on (str, consecutive)
# femurs: "SPXX-YY" XX={02-05}, YY={01-05}

#python -u esrgan_bones_v5-0.py 5 12 1e-4 1e-3 1 6e-3 0 "SP01-01" "SP01-02" "SP01-03" "SP01-04" "SP01-05"
#python -u esrgan_bones_v5-0.py 5 12 1e-4 1e-3 1 6e-3 0 "SP02-01" "SP02-02" "SP02-03" "SP02-04" "SP02-05"
#python -u esrgan_bones_v5-0.py 5 12 1e-4 1e-3 1 6e-3 0 "SP03-01" "SP03-02" "SP03-03" "SP03-04" "SP03-05"
#python -u esrgan_bones_v5-0.py 1 12 5e-5 1e-3 1 6e-3 0 "SP04-01" "SP04-02" "SP04-03" "SP04-04" "SP04-05"
python -u esrgan_bones_v5-0.py 1 12 5e-6 1e-3 1 6e-3 0 "SP05-02" "SP05-01" "SP05-03" "SP05-04" "SP05-05"
