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
        self.data = self.data[self.data['Error Flag'] != 0].reset_index()
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

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs,save_dir, checkpoint_path):
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print("No checkpoint found, starting training from scratch.")
    
    metrics = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, leave=False, desc="Training Batches"):
            optimizer.zero_grad()
            labels = batch['sentence_labels'].type(torch.float)
            preds = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(preds, labels)

            if torch.isnan(loss):
                continue

            # Prevent NaN values in gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad[torch.isnan(param.grad)] = 0

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')
        
        acc, val_loss = evaluate(model, val_loader, criterion, save_dir)
        print(f'Epoch {epoch+1}/{num_epochs}, Task 2 Accuracy: {acc}, Validation Loss: {val_loss}')

        metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "task2_acc": acc,
            "val_loss": val_loss
        })
        
        if epoch%10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    
    metrics_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(metrics_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "task2_acc", "val_loss"])
        writer.writeheader()
        for data in metrics:
            writer.writerow(data)
    print(f"Training metrics saved to {metrics_path}")

def evaluate(model, val_loader, criterion, save_dir):
    gold_sentence = []
    task2_preds = []
    
    model.eval()
    acc = 0
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc="Validation Batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask']
            labels = batch['sentence_labels'].type(torch.float)

            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            val_loss += loss.item()

            task2_pred = torch.argmax(preds, dim=1).cpu()
            sentence_labels = torch.argmax(labels, dim=1).cpu()
            acc += compute_accuracy(task2_pred, sentence_labels)
            gold_sentence.append(sentence_labels)
            task2_preds.append(task2_pred)
            
    output = {
        "gold_sentence": [tensor.numpy() for tensor in gold_sentence],
        "task2_preds": [tensor.numpy() for tensor in task2_preds]
    }

    with open(os.path.join(save_dir,'output.pkl'), 'wb') as file:
        pickle.dump(output, file)

    return acc / len(val_loader), val_loss / len(val_loader)

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
    parser.add_argument('--pre_m', type = str, default = 'emilyalsentzer/Bio_ClinicalBERT', help = 'pre-trained model name')
    parser.add_argument('--epoch', type = int, default = 5, help = 'epoch')
    parser.add_argument('--lr',type = float, default = '1e-5', help = 'learning rate')
    parser.add_argument('--num', type = int, default = 2000, help = 'MIMIC_sample_num')
    parser.add_argument('--dataset', type = str, default = 'UW', help = 'dataset name')
    parser.add_argument('--option', type = int, default = 3000, help = 'number of enhanced files')
    parser.add_argument('--bs',  type = int, default = 8, help = 'batch size')
    parser.add_argument('--regu',  type = float, default = 1e-5, help = 'batch size')
    parser.add_argument('--checkpoint_path', default = None, help = 'checkpoint_path')
    args = parser.parse_args()


    model_name = args.pre_m
    dataset = args.dataset
    learning_rate = args.lr
    MIMIC_sample_num = args.num
    epoch = args.epoch
    option = args.option
    batch_size = args.bs
    regu = args.regu
    checkpoint_path = args.checkpoint_path

    print("Dataset:", args.dataset)
    print("Learning Rate:", args.lr)
    print("MIMIC Sample Number:", args.num)
    print("Epoch:", args.epoch)
    print("Option:", args.option)
    print("Batch Size:", args.bs)
    print("Regularization:", args.regu)



    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if dataset == 'UW':
        train_data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
        train_df = pd.read_csv(os.path.join(train_data_dir, 'MEDIQA-CORR-2024-MS-TrainingData.csv'))
        train_dataset = EHRDataset(train_df,tokenizer)
        max_index = 41
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
    
    save_root = f'./models/task2/{model_name}'
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
        
    save_dir = os.path.join(save_root,f'dataset_{dataset}_epoch_{epoch}_learning_rate_{learning_rate}_regu_{regu}')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
    val_df = pd.read_csv(os.path.join(val_data_dir, 'MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv'),index_col = 0)

    
    val_dataset = EHRDataset(val_df,
                             tokenizer = tokenizer,
                             num_classes = max_index)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    model = BertErrorSentenceDetector(max_index,model_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=regu)
    
    train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epoch, save_dir, checkpoint_path = checkpoint_path)
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pth'))
        
if __name__ == '__main__':
    main()
  
# data_dir = './Data/UW/Feb_1_2024_MS_Train_Val_Datasets'
# train_df = pd.read_csv(os.path.join(data_dir, 'MEDIQA-CORR-2024-MS-TrainingData.csv'),index_col = 0)
# val_df = pd.read_csv(os.path.join(data_dir, 'MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv'),index_col = 0)
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# train_dataset = EHRDataset(train_df,tokenizer)
# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_dataset = EHRDataset(val_df,tokenizer)
# val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# loss_fn = nn.CrossEntropyLoss()
# model = BertErrorSentenceDetector()
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-5)
# train(model, train_dataloader, val_dataloader, loss_fn, optimizer, 5)
# torch.save(model.state_dict(), './models/test_task2.pth')