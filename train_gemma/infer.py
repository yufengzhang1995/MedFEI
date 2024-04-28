from unsloth import FastLanguageModel
import torch
from process_data import *
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

CHECKPOINT = "/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/checkpoint/baseline_epoch1_lr2e-3_lora_gemma7b"
TEST_SET = "/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/corrupt_files/uwformat_sampled_test.csv"
PRED = "/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/uwformat_sampled_test.baseline.pred.csv"

# load model and generate
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CHECKPOINT, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

df = pd.read_csv(TEST_SET)
df['prompts'] = df['Sentences'].apply(lambda x: f"Clinical Report: {x}\nAnalysis: ")
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

flags = []
idxes = []
ents = []
sents = []

for i in df['prompts']:
    inputs = tokenizer([i], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    pred = tokenizer.batch_decode(outputs)[0]
    if 'There is a factual error' in pred:
        pred_flag = 1
        try:
            pred_idx = re.search(r'There is a factual error in sentence ([0-9]+?)', pred).group(1)
        except:
            pred_idx = -1
        try:
            pred_ent = re.search(r'the error entity is "(.+?)"', pred).group(1)
        except:
            pred_ent = ''
        try:
            pred_corrected_sent = re.search(r'The corrected sentence is: (.+)', pred).group(1)
        except:
            pred_corrected_sent = ''
    else:
        pred_flag = 0
        pred_idx = -1
        pred_ent = ''
        pred_corrected_sent = ''
    flags.append(pred_flag)
    idxes.append(pred_idx)
    ents.append(pred_ent)
    sents.append(pred_corrected_sent)
df['pred_flags'] = flags
df['pred_error_sentence_id'] = idxes
df['pred_ents'] = ents
df['pred_corrected_sentence'] = sents
df.to_csv(PRED, index=False)