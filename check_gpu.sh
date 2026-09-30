#!/bin/bash
#SBATCH --job-name=check_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --time=00:01:00
#SBATCH --output=nvidia_smi_%j.log

nvidia-smi