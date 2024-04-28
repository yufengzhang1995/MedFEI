import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
import argparse
import csv

def handle_MIMIC_data(raw_df_2, sample_num = 2000):
    raw_df_2 = raw_df_2[(raw_df_2['correct_sentence']!='ISINF') & (raw_df_2['corrupted_sentence']!='ISINF')]
    raw_df_2 = raw_df_2[raw_df_2.apply(lambda x: x['corrupted_sentence']!=x['correct_sentence'], axis=1)]
    raw_df_2['sentence_id'] = pd.to_numeric(raw_df_2['sentence_id'], errors='coerce')
    max_index =  raw_df_2['sentence_id'].max() + 1
    raw_df_2_processed = raw_df_2[['correct_summary','corrupted_summary', 'correct_sentence', 'corrupted_sentence','sentence_id']]
    raw_df_2_processed = raw_df_2_processed.sample(sample_num)
    # correct 
    train_df_2 = raw_df_2_processed[['correct_summary']]
    train_df_2 = train_df_2.copy()
    train_df_2.loc[:,'Error Flag'] = 0
    train_df_2.loc[:,'Error Sentence ID'] = -1
    train_df_2.rename(columns={'correct_summary': 'Sentences'}, inplace=True)
    # corrupt
    error_sub = raw_df_2_processed[['corrupted_summary','sentence_id']].copy()
    error_sub.loc[:,'Error Flag'] = 1
    error_sub.rename(columns={'corrupted_summary': 'Sentences','sentence_id': 'Error Sentence ID'}, inplace=True)
    # concat
    train_df_2 = pd.concat([train_df_2,error_sub],axis = 0)
    train_df_2 = train_df_2.reset_index()
    num_classes = max(max_index,40) 
    return train_df_2,int(num_classes)

class EHRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, num_classes = 41, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe[dataframe['Error Flag'] != 0].reset_index()
        self.text = self.data['Sentences']
        self.binary_labels = self.data['Error Flag']
        self.sentence_labels = self.data['Error Sentence ID']
        self.max_length = max_length
        self.num_classes = num_classes
    def __len__(self):
        return len(self.text)
    def get_text(self,idx):
        text = str(self.text[idx])
        text = " ".join(text.split())
        return text
    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True
        )
        binary_label = self.binary_labels[idx]
        sentence_label = self.sentence_labels[idx]
        if binary_label == 0:
            sentence_label = torch.zeros(self.num_classes)
        else:
            sentence_label = F.one_hot(torch.tensor(sentence_label, dtype=torch.long), num_classes=self.num_classes)
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long).to(device),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device),
            'binary_labels': torch.tensor(binary_label, dtype=torch.float).to(device),
            'sentence_labels': sentence_label.to(device)
            
        }
        
   

train_data_dir = './Data/MIMIC'
model_name = "bert-base-uncased"
raw_df = pd.read_csv(os.path.join(train_data_dir, 'merged_corrupted_mimic.csv'))
# raw_df = pd.read_csv(os.path.join(train_data_dir, f'merged_corrupted_mimic_enhanced_3000.csv')) # column ['correct_summary', 'corrupted_summary', 'correct_sentence','corrupted_sentence', 'sentence_id']
# check the dimension of raw_df
print(raw_df.shape)
# check the first 5 rows of raw_df
print(raw_df.head())
# check the column names of raw_df
print(raw_df.columns)

MIMIC_sample_num = 2000
train_df, max_index = handle_MIMIC_data(raw_df, sample_num = MIMIC_sample_num)
print(train_df.shape)
print(train_df.head())
print(train_df.columns)

max_index
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = EHRDataset(train_df,
                        tokenizer = tokenizer,
                        num_classes = max_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size=32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
val_df = pd.read_csv(os.path.join(val_data_dir, 'MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv'),index_col = 0)
val_dataset = EHRDataset(val_df,tokenizer = tokenizer,num_classes = max_index)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Error show up
for batch in tqdm(train_dataloader, leave=False, desc="Training Batches"):
    print(batch)
            
