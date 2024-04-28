#!/bin/bash

#SBATCH --account=eecs598s16w24_class
#SBATCH --mail-user=jiacong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=spgpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=1000m
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --job-name=bert_task1

python3 codes/bert_task1_jiacong.py --dataset 'UW' --lr 1e-5 --epoch 70 --regu 0.01 --pre_m 'bert-base-uncased'
