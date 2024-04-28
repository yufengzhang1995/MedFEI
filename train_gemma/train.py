from unsloth import FastLanguageModel
import torch
from process_data import *
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import random

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

N_MIMIC = 1000
N_EPOCH = 2
LR = 2e-4
FRAC = 0.5
OUT_DIR = "/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598"
TRAIN_SET = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/corrupt_files/merged_corrupted_mimic_enhanced.n1000.csv'
TRAIN_SET2 = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-TrainingData.csv'
TEST_SET = '/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv'
PRED = f'/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/output/datasize/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.pred.uw_baseline_epoch{N_EPOCH}_lr{LR}.csv'
CHECKPOINT = f"/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/checkpoint/uw_baseline_epoch{N_EPOCH}_lr{LR}_lora_gemma7b"

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
] # More models at https://huggingface.co/unsloth

# load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    # model_name = "/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/checkpoint/mimic_n9000_epoch1_lr2e-4_lora_gemma7b"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
## add lora adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# process data
# d = load_local_data(TRAIN_SET2, EOS_TOKEN) # use UW dataset
# d['text'] = random.sample(d['text'], int(len(d['text'])*FRAC))

d = load_local_mimic_data(TRAIN_SET, EOS_TOKEN, n=N_MIMIC) # use MIMIC dataset
d['text'] += load_local_data(TRAIN_SET2, EOS_TOKEN)['text'] # combine with UW dataset

dataset = Dataset.from_dict(d)

# train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 60,
        num_train_epochs=N_EPOCH,
        learning_rate = LR,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        # output_dir = "outputs",
        output_dir = OUT_DIR
    ),
)
trainer_stats = trainer.train()

# inference
# evaluation
df = pd.read_csv(TEST_SET)
df['prompts'] = df['Sentences'].apply(lambda x: f"Clinical Report: {x}\nAnalysis: ")
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

flags = []
idxes = []
ents = []
sents = []
preds = []

for i in df['prompts']:
    inputs = tokenizer([i], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    pred = tokenizer.batch_decode(outputs)[0]
    if 'There is a factual error' in pred:
        pred_flag = 1
        try:
            pred_idx = re.search(r'There is a factual error in sentence ([0-9]+)', pred).group(1)
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
    preds.append(pred)
df['pred_flags'] = flags
df['pred_error_sentence_id'] = idxes
df['pred_ents'] = ents
df['pred_corrected_sentence'] = sents
df['pred'] = preds
df.to_csv(PRED, index=False)

# save model
model.save_pretrained(CHECKPOINT) # Local saving

# # load model and generate
# from unsloth import FastLanguageModel
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = CHECKPOINT, # YOUR MODEL YOU USED FOR TRAINING
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
# )
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# # alpaca_prompt = You MUST copy from above!

# inputs = tokenizer(
# [
#     "test input: "
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
# tokenizer.batch_decode(outputs)