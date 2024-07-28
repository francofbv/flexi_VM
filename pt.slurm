#!/bin/bash
#SBATCH --constraint=gpu80
#SBATCH --time=3-00:00:00
#SBATCH --job-name=finetune
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -n12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user vidalbfranco@gmail.com

export OMP_NUM_THREADS=1

JOB_NAME='finetune'
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export DECORD_EOF_RETRY_MAX=1310720
echo $DECORD_EOF_RETRY_MAX

nvidia-smi

#python3 pretrain.py \

#./exp_pt/videomamba_middle_5m/config.py \
python3 pretrain.py ./exp_pt/videomamba_middle_5m/config.py

