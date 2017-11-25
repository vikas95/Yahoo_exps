#!/bin/bash
#BSUB -n 16
#BSUB -R gpu
#BSUB -q "standard"
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -x
bsub -R gpu -Is bash
source ~/venv/bin/activate
module load python/3.5.2
module load cuda/8.0.61
python LSTM_POS.py
