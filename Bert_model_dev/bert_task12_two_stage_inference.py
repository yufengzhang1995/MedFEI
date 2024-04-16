"""
Author: Yufeng
2024/04/13

"""
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

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class BertBinaryClassifier(nn.Module):
    def __init__(self, 
                 pretrained_model_name = "emilyalsentzer/Bio_ClinicalBERT"):
        super(BertBinaryClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, 
                            attention_mask = attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class BertErrorSentenceDetector(nn.Module):
    def __init__(self, 
                 max_sentence_id = 40,
                 pretrained_model_name = "emilyalsentzer/Bio_ClinicalBERT",):
        super(BertErrorSentenceDetector, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, max_sentence_id)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, 
                            attention_mask = attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        probs = F.softmax(logits,dim = 1)
        return probs
    
class EHRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, num_classes = 41, max_length=512):

        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['Sentences']
        self.binary_labels = dataframe['Error Flag']
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
        



def inference(model1, model2,val_loader, save_dir):
    
    gold_binary = []
    gold_sentence = []
    
    task1_preds = []
    task2_preds = []
    
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc="Validation Batches"):
            binary_labels = batch['binary_labels'].type(torch.float)
            sentence_labels = batch['sentence_labels'].type(torch.float)

            pred_binary = model1(batch['input_ids'], batch['attention_mask'])
            pred_sentence  = model2(batch['input_ids'], batch['attention_mask'])
            
            task1_pred = torch.argmax(pred_binary, dim=1).cpu()
            gold_binary.append(binary_labels.cpu())
            task1_preds.append(task1_pred)

            task2_pred = torch.argmax(pred_sentence, dim=1).cpu()
            sentence_labels = torch.argmax(sentence_labels, dim=1).cpu()
            gold_sentence.append(sentence_labels)
            task2_preds.append(task2_pred)


    output = {
        "gold_binary": [tensor.numpy() for tensor in gold_binary],
        "gold_sentence": [tensor.numpy() for tensor in gold_sentence],
        "task1_preds": [tensor.numpy() for tensor in task1_preds],
        "task2_preds": [tensor.numpy() for tensor in task2_preds]
    }

    with open(os.path.join(save_dir,'output.pkl'), 'wb') as file:
        pickle.dump(output, file)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'Bert model training')
    parser.add_argument('--checkpoint1', type = str, default = 5, help = 'epoch')
    parser.add_argument('--checkpoint2',type = str, default = '1e-5', help = 'learning rate')
    args = parser.parse_args()
    checkpoint1 = args.checkpoint1
    checkpoint2 = args.checkpoint2
    save_dir = './outputs/stage1_stage2_UW'
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    
    val_data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
    val_df = pd.read_csv(os.path.join(val_data_dir, 'MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv'),index_col = 0)

    max_index = 42
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    val_dataset = EHRDataset(val_df,
                             tokenizer = tokenizer,
                             num_classes = max_index)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model1 = BertBinaryClassifier(pretrained_model_name = 'bert-base-uncased')
    state_dict = torch.load(checkpoint1, map_location='cpu')
    model1.load_state_dict(state_dict)   

    model2 = BertErrorSentenceDetector(max_sentence_id = max_index -1,pretrained_model_name = 'bert-base-uncased')
    state_dict = torch.load(checkpoint2, map_location='cpu')
    model2.load_state_dict(state_dict)   
    
    inference(model1, model2,val_dataloader, save_dir)

        
if __name__ == '__main__':
    main()