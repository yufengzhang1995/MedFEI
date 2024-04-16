"""
Author: Yufeng Zhang
April 6th, 2024
"""

# load fuzzy matched data
discharge = pd.read_csv('./data_temp/discharge_diagcat.csv',index_col = 0)
discharge['phecode'] = discharge['phecode'].astype(str)

# load phecode
ICD_phecode = pd.read_csv(os.path.join('./data_temp','Phecode_map_icd10cm_beta.csv'),encoding = "ISO-8859-1")
ICD_phecode['phecode'] = ICD_phecode['phecode'].astype(str)

# Subset fuzzy matched dataframe
# **based on the phecode, only keep those primary diagsnosis are considered certain disease** # 
disease_categories = ['circulatory system',
                      'congenital anomalies',
                      'dermatologic',
                      'digestive',
                      'endocrine/metabolic',
                      'genitourinary',
                      'hematopoietic',
                      'injuries & poisonings','mental disorders',
                      'musculoskeletal','neoplasms',
                      'neurological',
                      'pregnancy complications','respiratory','sense organs',
                      'symptoms']

import pandas as pd
import os, re, time
import numpy as np
from openai import OpenAI
client = OpenAI(api_key="")

def extract_icd_code(text):
    icd_pattern = re.compile(r"[A-TV-Z][0-9][0-9](?:\.[0-9A-Z]{1,4})?")
    if re.search(icd_pattern, text):
        matched = re.search(icd_pattern, text)[0]
        return matched
    else:
        return np.nan
    
def gen_gpt_icd(text):
    response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
    {"role": "system", "content": "You are an biomedical AI assitant"},
    {"role": "user", "content": f" Please provide ONLY the ICD-10 code of the diagnosis. No more than 10 characters{text}"},
  ]
    )
    return response.choices[0].message.content

def process_batch(dataframe, start_index, batch_size):
    end_index = min(start_index + batch_size, len(dataframe))
    for index, row in dataframe.iloc[start_index:end_index].iterrows():
        try:
            icd_code = gen_gpt_icd(row['primary_diagnosis'])
            dataframe.at[index, 'gpt_icd_10'] = extract_icd_code(icd_code)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    return dataframe

for category in disease_categories:
    if category == 'endocrine/metabolic':
        save_name = 'endocrine'
    if category == 'injuries & poisonings':
        save_name = 'injuries'
    else:
        save_name = category
    print(f'Start extract icd-10 code using gpt for {category}.')
    partial_file_path = f'./other_data_temp/discharge_{save_name}_partial.csv'

    if os.path.exists(partial_file_path):
        if category in ['genitourinary','hematopoietic']:
            discharge_category = pd.read_csv(partial_file_path)
            print(discharge_category.index[0])
        else:
            discharge_category = pd.read_csv(partial_file_path,index_col = 0)
        if discharge_category['gpt_icd_10'].isna().any():
            last_valid = discharge_category['gpt_icd_10'].last_valid_index()
            start_index = last_valid + 1 if last_valid is not None else 0
        else:
            start_index = len(discharge_category)
        print(f'Partial file found for {category}. Resuming from saved start index {start_index}')
    else:
        discharge_category = discharge[discharge.phecode.isin(ICD_phecode[ICD_phecode.exclude_name == category].phecode.unique())]
        discharge_category = pd.merge(discharge_category, ICD_code_str, on=['icd10cm'], how='left')
        if 'gpt_icd_10' not in discharge_category.columns:
            discharge_category['gpt_icd_10'] = None
        start_index = 0
        print(f'No Partial file found for {category}. Start from scratch!')
    batch_size = 100 
    target_size = 3000
    print(f'The batch size is {batch_size}')
    
    while start_index < target_size:
        discharge_category = process_batch(discharge_category, start_index, batch_size)
        start_index += batch_size
        # Save progress
        discharge_category.to_csv(partial_file_path, index=False)
        print(f"Processed and saved up to index {start_index} for {category}")

    discharge_category.to_csv(f'./other_data_temp/discharge_{save_name}_final.csv', index=False)
    print(f'Completed processing. Saved to ./other_data_temp/discharge_{save_name}_final.csv')
    del discharge_category
