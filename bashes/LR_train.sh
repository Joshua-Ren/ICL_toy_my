#!/bin/bash
#SBATCH --account=rrg-dsuth
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=10000M               # memory per node
#SBATCH --time=1-10:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 
#SBATCH --job-name=icl_toy


# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_icl/bin/activate

# 3. Copy your dataset on the compute node
#cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

cd /home/joshua52/projects/def-dsuth/joshua52/ICL_toy_my

srun python /home/joshua52/projects/def-dsuth/joshua52/ICL_toy_my/train.py \
--config_file LR_train