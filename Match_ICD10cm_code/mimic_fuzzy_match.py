import numpy as np
import pandas as pd
import os,sys
import re
from thefuzz import fuzz

dir_root = '{}'
discharge = pd.read_csv(os.path.join(dir_root,'discharge.csv'))


discharge = discharge[['note_id', 'subject_id', 'hadm_id', 'text']]

def condense_text(text):
    text = re.sub(r'\=+', '', text)
    text = re.sub(r'\n\n', '. ', text) 
    text = re.sub(r'(?<!:)\n', ', ', text) 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'\-|_', '', text) 
    return text.strip()
def find_diagnosis(note):
    diagnosis_pattern1 = re.compile(r"Discharge Diagnosis:\n(.*?)\n\s+\nDischarge Condition:", re.DOTALL)
    primary_diagnosis_pattern1 = r'Primary:\s*([^,]+)'
    primary_diagnosis_pattern2 = r'PRIMARY DIAGNOSIS[:.]\s*([^,\.]+)'

    matched = re.search(diagnosis_pattern1, note)
    if matched:
        text = matched.group(1).strip()

        # Assuming condense_text is a predefined function
        text = condense_text(text)

        primary_diagnosis = None
        if re.search(primary_diagnosis_pattern1, text, re.IGNORECASE):
            primary_diagnosis = re.search(primary_diagnosis_pattern1, text, re.IGNORECASE)
        elif re.search(primary_diagnosis_pattern2, text, re.IGNORECASE):
            primary_diagnosis = re.search(primary_diagnosis_pattern2, text, re.IGNORECASE)

        primary_diagnosis_match = primary_diagnosis.group(1) if primary_diagnosis else text
        return primary_diagnosis_match
    else:
        return np.nan
    
discharge['primary_diagnosis'] = discharge['text'].apply(find_diagnosis)

ICD_phecode = pd.read_csv(os.path.join('./Med_FEI/data_temp','Phecode_map_v1_2_icd10cm_beta.csv'),encoding = "ISO-8859-1")
ICD_phecode['phecode'] = ICD_phecode['phecode'].astype(str)
ICD_phecode_infection_ls = ICD_phecode[ICD_phecode.exclude_name == 'infectious diseases'].phecode.unique()

def find_corre_phecode(ICD_phecode, target_string):
    if pd.isna(target_string):
        return np.nan, np.nan  # Return a tuple of NaNs

    fuzzy_ratios = []
    for code in ICD_phecode['icd10cm_str']:
        ratio = fuzz.ratio(target_string, code)
        fuzzy_ratios.append(ratio)
    target_index = np.argmax(fuzzy_ratios)
    target_ICD_phecode = ICD_phecode.iloc[target_index][['icd10cm', 'phecode']]
    return target_ICD_phecode['icd10cm'], target_ICD_phecode['phecode']

results = discharge['primary_diagnosis'].apply(lambda x: find_corre_phecode(ICD_phecode, x))
discharge['icd10cm'], discharge['phecode'] = zip(*results)
discharge.to_csv('./Med_FEI/data_temp/discharge_diagcat.csv')
