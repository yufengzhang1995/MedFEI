import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc 
import matplotlib.pyplot as plt 

def evaluate(y_true, y_pred, multiclass=False):
    # Confusion Matrix 
    # cm = confusion_matrix(y_true, y_pred) 
    # Accuracy 
    accuracy = accuracy_score(y_true, y_pred) 
    # Precision 
    if not multiclass:
        precision = precision_score(y_true, y_pred) 
    else:
        precision = {'micro': precision_score(y_pred, y_pred, average='micro'),
        'macro': precision_score(y_pred, y_pred, average='macro')}
    # Recall 
    if not multiclass:
        recall = recall_score(y_true, y_pred) 
    else:
        recall = {'micro': recall_score(y_true, y_pred, average='micro'),
        'macro': recall_score(y_true, y_pred, average='macro') }
    # F1-Score 
    if not multiclass:
        f1 = f1_score(y_true, y_pred) 
    else:
        f1 = {'micro': f1_score(y_true, y_pred, average='micro'),
        'macro': f1_score(y_true, y_pred, average='macro') }
    # # ROC Curve and AUC 
    # if not multiclass:
    #     fpr, tpr, thresholds = roc_curve(y_true, y_pred) 
    #     roc_auc = auc(fpr, tpr) 
    # else:
    #     fpr = 'Not applicable'
    #     tpr = 'Not applicable'
    #     thresholds = None
    #     roc_auc = None

    # print("Confusion Matrix:") 
    # print(cm) 
    print("Accuracy:", accuracy) 
    print("Precision:", precision) 
    print("Recall:", recall) 
    print("F1-Score:", f1) 
    # print("ROC AUC:", roc_auc) 

    # return fpr, tpr, thresholds


PRED = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/epoch/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.uw_baseline_epoch1.csv'
# PRED = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/train_size/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.mimic_only_n7000_epoch1_lr2e-4.csv'
# PRED = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/uwformat_sampled_test.pred.csv'
# PRED = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/uwformat_sampled_test.baseline.pred.csv'
# PRED = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/fine-tune/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.n9000_epoch1_lr2e-4.csv'

# load data
df = pd.read_csv(PRED)

# Factual error detection (binary classification)
labels = df['Error Flag'].to_list()
preds = df['pred_flags'].to_list()
evaluate(labels, preds)

# Error sentence detection (multi-class classification)
labels = df['Error Sentence ID'].to_list()
preds = df['pred_error_sentence_id'].to_list()
evaluate(labels, preds, multiclass=True)
