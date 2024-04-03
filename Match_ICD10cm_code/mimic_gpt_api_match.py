
import numpy as np
import pandas as pd
import os,sys
import re
import time
import ast

# load phecode
ICD_phecode = pd.read_csv(os.path.join('./Med_FEI/data_temp','Phecode_map_v1_2_icd10cm_beta.csv'),encoding = "ISO-8859-1")
ICD_phecode['phecode'] = ICD_phecode['phecode'].astype(str)


# load infection
discharge_infect = pd.read_csv('./Med_FEI/data_temp/discharge_infect.csv',index_col = 0)

# generate icd code using gpt
from openai import OpenAI
client = OpenAI(api_key="")
def gen_gpt_icd(text):
    response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
    {"role": "system", "content": "You are an biomedical AI assitant"},
    {"role": "user", "content": f" Please provide ONLY the ICD-10 code of the diagnosis. The provided ICD code should be correlated with infection. {text}"},
  ]
    )
    return response.choices[0].message.content

# post-process text
def extract_icd_code(text):
    icd_pattern = re.compile(r" [A-TV-Z][0-9][0-9](?:\.[0-9A-Z]{1,4})?")
    if re.search(icd_pattern, text):
        matched = re.search(icd_pattern, text)[0]
        return matched
    else:
        return np.nan
    
discharge_infect['gpt_icd_10'] = discharge_infect['primary_diagnosis'].apply(gen_gpt_icd)
discharge_infect['gpt_icd_10'] = discharge_infect['gpt_icd_10'].apply(extract_icd_code)
discharge_infect.to_csv('./Med_FEI/data_temp/discharge_infect.csv')
print('save discharge_infect with corrected ICD code and phecode to ./Med_FEI/data_temp/discharge_infect.csv')