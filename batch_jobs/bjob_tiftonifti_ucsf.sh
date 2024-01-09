#!/bin/sh
#BSUB -q gpuv100
#BSUB -J tif_to_nifty
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o /work3/soeba/HALOS/experiments/tifstack_to_nifty/%J.out
#BSUB -e /work3/soeba/HALOS/experiments/tifstack_to_nifty/%J.err

cd ..

source init.sh

python -u tiftonifty_ucsf.py "SP04-01"
python -u tiftonifty_ucsf.py "SP04-02"
python -u tiftonifty_ucsf.py "SP04-03"
python -u tiftonifty_ucsf.py "SP04-04"
python -u tiftonifty_ucsf.py "SP04-05"
python -u tiftonifty_ucsf.py "SP05-01"
python -u tiftonifty_ucsf.py "SP05-02"
python -u tiftonifty_ucsf.py "SP05-03"
python -u tiftonifty_ucsf.py "SP05-04"
python -u tiftonifty_ucsf.py "SP05-05"



