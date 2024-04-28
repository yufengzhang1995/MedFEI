import pandas as pd
import difflib
import re


# EOS_TOKEN = '<eos>'

def tokenize(s):
    return re.split('\s+', s)
def untokenize(ts):
    return ' '.join(ts)

def equalize(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    res1 = []
    res2 = []
    prev = difflib.Match(0,0,0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if (prev.a + prev.size != match.a):
            for i in range(prev.a + prev.size, match.a):
                res2 += ['_' * len(l1[i])]
            res1 += l1[prev.a + prev.size:match.a]
        if (prev.b + prev.size != match.b):
            for i in range(prev.b + prev.size, match.b):
                res1 += ['_' * len(l2[i])]
            res2 += l2[prev.b + prev.size:match.b]
        res1 += l1[match.a:match.a+match.size]
        res2 += l2[match.b:match.b+match.size]
        prev = match
    return untokenize(res1), untokenize(res2)

def get_diff(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    prev = difflib.Match(0,0,0)
    max_len_unmatched = 0
    error = ()
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if (prev.a + prev.size != match.a) or (prev.b + prev.size != match.b):
            len_unmatched = max(match.a - (prev.a + prev.size), match.b - (prev.b + prev.size))
            if len_unmatched > max_len_unmatched:
                a_text = l1[prev.a + prev.size:match.a]
                b_text = l2[prev.b + prev.size:match.b]
                error = (untokenize(a_text), untokenize(b_text))
                max_len_unmatched = len_unmatched
        prev = match
    return error

def remove_pref(text):
    return re.sub(r'^[0-9+]\.?', '', text).strip()

def generate_prompt(input, label, error_id, diff, corrected, eos):
  prompt = f"""Clinical Report: {input}"""
  if label == 1:
    prompt += f"\nAnalysis: There is a factual error in sentence {error_id} of this report. In this sentence, the error entity is \"{diff[0]}\", which should be \"{diff[1]}\". The corrected sentence is: {corrected}"
  else:
    prompt += "\nAnalysis: There is no factual error in this report."
  return prompt  + eos

def load_local_data(TRAIN_SET, eos):
  df = pd.read_csv(TRAIN_SET)
  df['diff_texts'] = df.fillna('').apply(lambda x: get_diff(x['Error Sentence'], x['Corrected Sentence']), axis=1)
  df['prompts'] = df.apply(lambda x: generate_prompt(x["Sentences"], x['Error Flag'], x["Error Sentence ID"], x["diff_texts"], x['Corrected Sentence'], eos), axis=1)
  dataset = { "text" : df['prompts'].to_list() }
  return dataset

def load_local_mimic_data(TRAIN_SET, eos, n=2000):
  df = pd.read_csv(TRAIN_SET).sample(n)
  df = df[(df['correct_sentence']!='ISINF') & (df['corrupted_sentence']!='ISINF')].dropna()
  df = df[df.apply(lambda x: x['corrupted_sentence']!=x['correct_sentence'], axis=1)]
#   sub = df.sample(n)
  sub = df
  sub['diff_texts'] = sub.fillna('').apply(lambda x: get_diff(x['corrupted_sentence'], x['correct_sentence']), axis=1)
  sub_correct = sub.apply(lambda x: generate_prompt(x["correct_summary"], 0, -1, (), '', eos), axis=1)
  sub_corrupt = sub.apply(lambda x: generate_prompt(x["corrupted_summary"], 1, x['sentence_id'], x['diff_texts'], remove_pref(x['correct_sentence']), eos), axis=1)
  p = sub_correct.to_list() + sub_corrupt.to_list()
  dataset = { "text" : p }
  return dataset

def generate_mimic_test():
    df = pd.read_csv('/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/corrupt_files/merged_corrupted_mimic.csv')
    df = df[df.apply(lambda x: x['corrupted_sentence']!=x['correct_sentence'], axis=1)]
    sub = df.sample(1000)

    sub_correct = sub[["correct_summary", 'correct_sentence']]
    sub_correct['Error Flag'] = 0
    sub_correct['Error Sentence ID'] = -1
    sub_correct.columns = ['Sentences', 'Correct Sentence', 'Error Flag', 'Error Sentence ID']

    sub_corrupt = sub[["corrupted_summary", 'correct_sentence', 'sentence_id']]
    sub_corrupt['Error Flag'] = 1
    sub_corrupt.columns = ['Sentences', 'Correct Sentence', 'Error Sentence ID', 'Error Flag']

    d = pd.concat([sub_correct, sub_corrupt])
    d.to_csv("/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/corrupt_files/uwformat_sampled_test.csv", index=False)