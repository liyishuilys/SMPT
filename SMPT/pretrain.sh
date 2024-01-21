#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -G, --gpus=1
#SBATCH --cpus-per-gpu=16
srun python -m paddle.distributed.launch --log_dir ./log2/ab pretrain.py  # ab, ab_ba, ab_ba_da
