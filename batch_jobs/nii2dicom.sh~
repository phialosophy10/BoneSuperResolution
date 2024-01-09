#!/bin/sh
#BSUB -q gpuv100
#BSUB -J vol_to_slices
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/vol_to_slices/%J.out
#BSUB -e /work3/soeba/HALOS/experiments/vol_to_slices/%J.err

cd ..

source init.sh

# args: 
# femur_no (str, {001,002,013,015,021,026,031,074,075,083,086,138,164,172})

python -u vol_to_slices10.py "001"
python -u vol_to_slices10.py "002"
python -u vol_to_slices10.py "013"
python -u vol_to_slices10.py "015"
python -u vol_to_slices10.py "021"
python -u vol_to_slices10.py "026"
python -u vol_to_slices10.py "031"
python -u vol_to_slices10.py "074"
python -u vol_to_slices10.py "075"
python -u vol_to_slices10.py "083"
python -u vol_to_slices10.py "086"
python -u vol_to_slices10.py "138"
python -u vol_to_slices10.py "164"
python -u vol_to_slices10.py "172"


