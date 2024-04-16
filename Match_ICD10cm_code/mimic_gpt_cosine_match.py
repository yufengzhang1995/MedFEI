"""
Author: Yufeng
2024/04/04

"""



import numpy as np
import pandas as pd
import os,sys
import re
import time
import ast

# load fuzzy matched data
discharge = pd.read_csv('./Med_FEI/data_temp/discharge_diagcat.csv',index_col = 0)
discharge['phecode'] = discharge['phecode'].astype(str)
# load phecode
ICD_phecode = pd.read_csv(os.path.join('./Med_FEI/data_temp','Phecode_map_v1_2_icd10cm_beta.csv'),encoding = "ISO-8859-1")
ICD_phecode['phecode'] = ICD_phecode['phecode'].astype(str)
ICD_phecode_infection_ls = ICD_phecode[ICD_phecode.exclude_name == 'infectious diseases'].phecode.unique()
# based on the phecode, only keep those primary diagsnosis are considered infectious disease
discharge_infect = discharge[discharge.phecode.isin(ICD_phecode_infection_ls)]
discharge_infect.to_csv('./Med_FEI/data_temp/discharge_infect.csv')
print('save discharge_infect to ./Med_FEI/data_temp/discharge_infect.csv')

# get embeddings for every potentially infectious disease-related diagnosis
from openai import OpenAI
client = OpenAI(api_key="")
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
start_time = time.time()
discharge_infect['primary_diagnosis_embedding'] = discharge_infect['primary_diagnosis'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
end_time = time.time()
total_time = end_time - start_time
print("Time taken to run the code: {:.2f} seconds".format(total_time))
print('save discharge_infect to ./Med_FEI/data_temp/discharge_infect.csv')

# read embneddings for every ground-truth infectious disease (ICD10)
ICD_phecode_infect = pd.read_csv('./Med_FEI/data_temp/ICD_phecode_infect.csv',index_col = 0)
infect_icd_embedding = ICD_phecode_infect['icd10_embedding'].tolist()  # Convert the column to a list of lists
infect_icd_embedding = [ast.literal_eval(i) for i in infect_icd_embedding]
infect_icd_embedding_matrix = np.zeros((len(infect_icd_embedding), 1536))
for i, embedding in enumerate(infect_icd_embedding):
    infect_icd_embedding_matrix[i, :] = embedding

diagnosis_embedding = discharge_infect['primary_diagnosis_embedding'].tolist()  
diagnosis_embedding_matrix = np.zeros((discharge_infect.shape[0], 1536))
for i, embedding in enumerate(diagnosis_embedding):
    diagnosis_embedding_matrix[i, :] = embedding
    
# construct cosine similarity matrix
similarity_matrix = diagnosis_embedding_matrix.dot(infect_icd_embedding_matrix.T)
infec_select_index = np.argmax(similarity_matrix,axis = 1)
np.save('./Med_FEI/data_temp/infec_select_index.npy', infec_select_index)
print('save selected index using GPT to ./Med_FEI/data_temp/infec_select_index.npy')


# 
discharge_infect.rename(columns={'icd10cm':'fuzz_icd10cm',
                                    'phecode': 'fuzz_phecode'}, inplace=True)
discharge_infect = discharge_infect.reset_index(drop=True)
selected_columns = ICD_phecode_infect.iloc[infec_select_index][['icd10cm','icd10cm_str','phecode']].reset_index(drop=True)
renamed_columns = selected_columns.rename(columns={'icd10cm': 'corrected_icd', 
                                                   'icd10cm_str': 'corrected_icd10str', 
                                                   'phecode': 'corrected_phecode'})
discharge_infect = discharge_infect.assign(**renamed_columns)
discharge_infect.to_csv('./Med_FEI/data_temp/discharge_infect.csv')
print('save discharge_infect with corrected ICD code and phecode to ./Med_FEI/data_temp/discharge_infect.csv')

