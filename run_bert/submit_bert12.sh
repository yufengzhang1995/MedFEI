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
#SBATCH --job-name=bert_t12

python3 codes/bert_task12_two_stage_inference_jiacong.py --save_dir 'outputs/stage1_stage2_UW_model' --checkpoint1 'models/task1/bert-base-uncased/dataset_UW_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_UW_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' 
python3 codes/bert_task12_two_stage_inference_jiacong.py --save_dir 'outputs/stage1_stage2_mimic_enhanced_model' --checkpoint1 'models/task1/bert-base-uncased/dataset_mimic_enhanced_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_mimic_enhanced_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' 
python3 codes/bert_task12_two_stage_inference_jiacong.py --save_dir 'outputs/stage1_stage2_mimic_enhanced2_model' --checkpoint1 'models/task1/bert-base-uncased/dataset_mimic_enhanced2_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_mimic_enhanced2_epoch_70_learning_rate_1e-05_regu_0.01/model.pth'
python3 codes/bert_task12_two_stage_inference_jiacong.py --save_dir 'outputs/stage1_stage2_combined_enhanced_model' --checkpoint1 'models/task1/bert-base-uncased/dataset_combined_enhanced_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_combined_enhanced_epoch_70_learning_rate_1e-05_regu_0.01/model.pth'
python3 codes/bert_task12_two_stage_inference_jiacong.py --save_dir 'outputs/stage1_stage2_combined_enhanced2_model' --checkpoint1 'models/task1/bert-base-uncased/dataset_combined_enhanced2_epoch_70_learning_rate_1e-05_regu_0.01/model.pth' --checkpoint2 'models/task2/bert-base-uncased/dataset_combined_enhanced2_epoch_70_learning_rate_1e-05_regu_0.01/model.pth'
