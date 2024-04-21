#!/bin/bash

#SBATCH --account=eecs598s16w24_class
#SBATCH --mail-user=jiacong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=9600m
#SBATCH --nodes=1
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --job-name=bert_t2

python3 codes/bert_task2.py --dataset 'UW' --lr 1e-5 --epoch 70 --regu 0.01 --pre_m 'bert-base-uncased' --checkpoint 'models/task2/bert-base-uncased/dataset_UW_epoch_70_learning_rate_1e-5_regu_0.01/model.pth'
