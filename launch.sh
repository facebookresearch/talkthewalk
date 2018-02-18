#!/bin/bash
## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=evaluate
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/private/home/harm/slurm_logs/%j.out

## filename for job standard error output (stderr)
#SBATCH --error=/private/home/harm/slurm_logs/%j.err

## partition name
#SBATCH --partition=learnfair

## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
module purge

# Load what we need
source ~/.bashrc
module load anaconda3
module load cuda/8.0
module load cudnn/v6.0
source activate pytorch_mpi
export PYTHONPATH=/private/home/harm/code/ParlAI:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TALKTHEWALK_DATADIR=/private/home/harm/code/talkthewalk/data
export TALKTHEWALK_EXPDIR=/private/home/harm/exp

### Section 3:
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
### This is going to run ntasks-per-node x nodes tasks with each
### task seeing all the GPUs on each each node. However I am using
### the wrapper.sh example I showed before so that each task only
### sees one GPU

srun --label /private/home/harm/.conda/envs/pytorch_mpi/bin/python predict_location_language.py --exp-name dialogue_a4 --num-steps 4 --condition-on-action --batch-sz 128 --num-epochs 20 --full-dialogue
