#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8000M               # memory (per node)
#SBATCH --time=0-23:59            # time (DD-HH:MM)

module load miniconda3
source activate pt
export TALKTHEWALK_EXPDIR=~/exp

python ~/talkthewalk/predict_location_discrete.py --goldstandard-features --masc --T 2 --num-epochs 200 --exp-name disc_T2_masc --cuda


