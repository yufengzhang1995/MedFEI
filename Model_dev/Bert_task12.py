"""
Author: Yufeng Zhang
Date: April 13th 2024

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

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class BertErrorSentenceDetector(nn.Module):
    def __init__(self, 
                 num_classes = 2,
                 max_sentence_id = 40,
                 pretrained_model_name = "emilyalsentzer/Bio_ClinicalBERT",):
        super(BertErrorSentenceDetector, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes + max_sentence_id)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, 
                            attention_mask = attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        binary_output = nn.Sigmoid()(logits[:,0])
        sentence_output = F.softmax(logits[:,1:],dim = 1)
        return binary_output, sentence_output
    
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
        
        
class EHRCombinedDataset(Dataset):
    def __init__(self, 
                 dataframe1, 
                 dataframe2, 
                 tokenizer, 
                 num_classes = 110, 
                 max_length = 512):
        np.random.seed(42)
        
        self.tokenizer = tokenizer
        # concatenate dataframe and then sample
        self.data = pd.concat([dataframe1,dataframe2],axis = 0)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
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

def compute_accuracy(pred, gt):
    return sum(pred == gt)/len(pred)

def train(model, train_loader, val_loader, criterion_binary,criterion_multiple, optimizer, num_epochs,save_dir):
    for epoch in range(num_epochs):
        model.train()
        
        train_loss = 0
        
        for batch in tqdm(train_loader, leave=False, desc="Training Batches"):
            
            optimizer.zero_grad()
            
            
            binary_labels = batch['binary_labels'].type(torch.float)
            sentence_labels = batch['sentence_labels'].type(torch.float)
            

            pred_binary, pred_sentence = model(batch['input_ids'], batch['attention_mask'])

            binary_loss = criterion_binary(pred_binary, binary_labels)
            sentence_loss = criterion_multiple(pred_sentence, sentence_labels)

            loss = binary_loss + sentence_loss

            # Prevent NaN values in gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.isnan(param.grad)] = 0

            loss.backward()
            optimizer.step()
            train_loss = train_loss + binary_loss.item() + sentence_loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')
        task1_acc, task2_acc, val_loss = evaluate(model, val_loader, criterion_binary, criterion_multiple, save_dir)
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Task 1 Accuracy: {task1_acc}, Task 2 Accuracy: {task2_acc}, Validation Loss: {val_loss}')

def evaluate(model, val_loader, criterion_binary, criterion_multiple,save_dir):
    
    gold_binary = []
    gold_sentence = []
    
    task1_preds = []
    task2_preds = []
    
    model.eval()
    task1_acc = 0
    task2_acc = 0
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc="Validation Batches"):
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask']

            binary_labels = batch['binary_labels'].type(torch.float)
            sentence_labels = batch['sentence_labels'].type(torch.float)

            pred_binary, pred_sentence = model(batch['input_ids'], batch['attention_mask'])
            
            binary_loss = criterion_binary(pred_binary, binary_labels)
            sentence_loss = criterion_multiple(pred_sentence, sentence_labels)
            
            val_loss = val_loss + binary_loss.item() + sentence_loss.item()

            

            task1_pred = (pred_binary > 0.5).cpu()
            task1_acc += compute_accuracy(task1_pred, binary_labels.cpu())
            gold_binary.append(binary_labels.cpu())
            task1_preds.append(task1_pred)

            task2_pred = torch.argmax(pred_sentence, dim=1).cpu()
            sentence_labels = torch.argmax(sentence_labels, dim=1).cpu()
            task2_acc += compute_accuracy(task2_pred, sentence_labels)
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
    
    return task1_acc / len(val_loader), task2_acc / len(val_loader), val_loss / len(val_loader)



def handle_MIMIC_data(raw_df_2, sample_num = 2000):
    
    raw_df_2 = raw_df_2[(raw_df_2['correct_sentence']!='ISINF') & (raw_df_2['corrupted_sentence']!='ISINF')]
    raw_df_2 = raw_df_2[raw_df_2.apply(lambda x: x['corrupted_sentence']!=x['correct_sentence'], axis=1)]
    
    max_index = raw_df_2['sentence_id'].max() + 1
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
    error_sub.rename(columns={'corrupted_summary': 'Sentences',
                             'sentence_id': 'Error Sentence ID'}, inplace=True)

    # concat
    train_df_2 = pd.concat([train_df_2,error_sub],axis = 0)
    train_df_2 = train_df_2.reset_index()
    num_classes = max(max_index,40) 
    return train_df_2,int(num_classes)







def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description = 'Bert model training')
    parser.add_argument('--epoch', type = int, default = 10, help = 'epoch')
    parser.add_argument('--lr',type = float, default = '1e-5', help = 'learning rate')
    parser.add_argument('--num', type = int, default = 2000, help = 'MIMIC_sample_num')
    parser.add_argument('--dataset', type = str, default = 'UW', help = 'dataset name')
    parser.add_argument('--option', type = int, default = 3000, help = 'number of enhanced files')
    parser.add_argument('--bs',  type = int, default = 8, help = 'batch size')
    args = parser.parse_args()



    dataset = args.dataset
    learning_rate = args.lr
    MIMIC_sample_num = args.num
    epoch = args.epoch
    option = args.option
    batch_size = args.bs

    print("Dataset:", args.dataset)
    print("Learning Rate:", args.lr)
    print("MIMIC Sample Number:", args.num)
    print("Epoch:", args.epoch)
    print("Option:", args.option)
    print("Batch Size:", args.bs)




# dataset = 'MIMIC'
# option = '3000'



# epoch_ls = [10,20,30]
# learning_rate_ls = [1e-5,1e-4,1e-3]
# MIMIC_sample_num = 2000
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    if dataset == 'UW':
        train_data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
        train_df = pd.read_csv(os.path.join(train_data_dir, 'MEDIQA-CORR-2024-MS-TrainingData.csv'))
        train_dataset = EHRDataset(train_df,tokenizer)

    elif dataset == 'MIMIC':
        train_data_dir = './Data/MIMIC'
        raw_df = pd.read_csv(os.path.join(train_data_dir, 'merged_corrupted_mimic.csv'))
        train_df, max_index = handle_MIMIC_data(raw_df, sample_num = MIMIC_sample_num)
        train_dataset = EHRDataset(train_df,
                                tokenizer = tokenizer,
                                num_classes = max_index)

    elif 'mimic_enhanced' in dataset:
        train_data_dir = './Data/MIMIC'
        raw_df = pd.read_csv(os.path.join(train_data_dir, f'merged_corrupted_mimic_enhanced_{option}.csv'))
        train_df, max_index = handle_MIMIC_data(raw_df, sample_num = MIMIC_sample_num)
        train_dataset = EHRDataset(train_df,
                                tokenizer = tokenizer,
                                num_classes = max_index)
        # print(train_dataset[0])
    
    elif dataset == 'combined':
        train_data_dir_1 = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
        train_data_dir_2 = './Data/MIMIC'
        train_df_1 = pd.read_csv(os.path.join(train_data_dir_1, 'MEDIQA-CORR-2024-MS-TrainingData.csv'))
        train_df_1 = train_df_1[['Sentences', 'Error Flag', 'Error Sentence ID']]
        
        raw_df_2 = pd.read_csv(os.path.join(train_data_dir_2, 'merged_corrupted_mimic.csv'))
        train_df_2, max_index = handle_MIMIC_data(raw_df_2, sample_num = MIMIC_sample_num)
        
        train_dataset = EHRCombinedDataset(train_df_1,
                                train_df_2,
                                tokenizer = tokenizer,
                                num_classes = max_index)
    
    elif dataset == 'combined_enhanced' and option == 3000:
        train_data_dir_1 = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
        train_data_dir_2 = './Data/MIMIC'
        train_df_1 = pd.read_csv(os.path.join(train_data_dir_1, 'MEDIQA-CORR-2024-MS-TrainingData.csv'))
        train_df_1 = train_df_1[['Sentences', 'Error Flag', 'Error Sentence ID']]
        
        raw_df_2 = pd.read_csv(os.path.join(train_data_dir_2, f'merged_corrupted_mimic_enhanced_{option}.csv'))
        train_df_2, max_index = handle_MIMIC_data(raw_df_2, sample_num = MIMIC_sample_num)
        train_dataset = EHRCombinedDataset(train_df_1,
                                train_df_2,
                                tokenizer = tokenizer,
                                num_classes = max_index)
    else:
        raise print("Dataset not found")
    
    print('Length of the dataset:',len(train_dataset))
    
    save_dir = f'./models/dataset_{dataset}_epoch_{epoch}_learning_rate_{learning_rate}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
    val_df = pd.read_csv(os.path.join(val_data_dir, 'MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv'),index_col = 0)

    
    val_dataset = EHRDataset(val_df,
                             tokenizer = tokenizer,
                             num_classes = max_index)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn_binary = nn.BCELoss()
    loss_fn_multiple = nn.CrossEntropyLoss()
    model = BertErrorSentenceDetector(max_sentence_id = max_index-1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    train(model, train_dataloader, val_dataloader, loss_fn_binary, loss_fn_multiple, optimizer, epoch,save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pth'))
        
if __name__ == '__main__':
    main()
