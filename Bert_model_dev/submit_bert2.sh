#!/bin/bash

#SBATCH --job-name=bert_task2
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=slurm_out/%A_%a.out
#SBATCH --account=
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=9600m
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1



python bert_task2.py --dataset 'UW' --lr 1e-5 --epoch 10 --regu 0.01 --pre_m 'bert-base-uncased' --checkpoint 'models/task2/bert-base-uncased/dataset_UW_epoch_20_learning_rate_1e-06_regu_0.01/model.pth'
python bert_task2.py --dataset 'combined_enhanced' --lr 1e-5 --epoch 20 --regu 0.01 --pre_m 'bert-base-uncased' --checkpoint 'models/task2/bert-base-uncased/dataset_combined_enhanced_epoch_20_learning_rate_1e-06_regu_0.01/model.pth'


python bert_task12_two_stage_inference.py --checkpoint1 'models/task1/bert-base-uncased/dataset_combined_enhanced_epoch_5_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_combined_enhanced_epoch_20_learning_rate_1e-06_regu_0.01/model.pth'
python bert_task12_two_stage_inference.py --checkpoint1 'models/task1/bert-base-uncased/dataset_UW_epoch_5_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_UW_epoch_10_learning_rate_1e-05_regu_0.01/model.pth'