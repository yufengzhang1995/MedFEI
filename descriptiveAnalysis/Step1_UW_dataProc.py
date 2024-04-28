


import os
import openai
import requests
import json
import pandas as pd
import numpy as np
import csv

print("Start the file processing")

openai.api_key = ('input_the_openai_api_key_here')

trainData_filePath = '/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/data/MEDIQA-CORR-2024-MS-TrainingData.csv'
output_filePath = '/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/Step1_MEDIQA-CORR-2024-MS-TrainingData_process2.csv'
data_example_path = '/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/data/trainData_example5.csv'

# The URL for the OpenAI API completion endpoint
url = 'https://api.openai.com/v1/chat/completions'

# The headers including the Authorization with your API key
headers = {'Authorization': f'Bearer {openai.api_key}','Content-Type': 'application/json'}

# read the UW data
sample_data = pd.read_csv(trainData_filePath)

# provide some examples for the model to understand the task
data_example = pd.read_csv('/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/data/trainData_example5_v2.csv')
data_example1 = data_example[['Text ID','Corrected_Text_index','Diagnosis','ICD10']]
data_example1 = data_example1.iloc[:5]

example_str = []
for i in range(len(data_example1)):
    row = data_example1.iloc[i]
    example_str.append(f"Example {i+1}: [Correct_Text_index: {row['Corrected_Text_index']}, Diagnosis: '{row['Diagnosis']}', ICD10: '{row['ICD10']}']")

### function to index the correct paragraph ### 
def index_sen(data_i):
    if pd.isna(data_i['Corrected Text']):
        data = {
            "model": "gpt-3.5-turbo",  # Assuming you are using a chat model like gpt-3.5-turbo
            "messages": [
                {"role": "user", "content": f"Please index the sentence in the paragraph and return with indexed sentences,{data_i['Text']}"},
            ]
        }
    else:
        data = {
            "model": "gpt-3.5-turbo",  # Assuming you are using a chat model like gpt-3.5-turbo
            "messages": [
                {"role": "user", "content": f"Please index the sentence in the paragraph and return with indexed sentences,{data_i['Corrected Text']}"},
            ]
        }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print(f"Index_sen Success:")
        text_index_i = response.json()['choices'][0]['message']['content']
    else:
        print(f"Index_sen Error:")
        print(response.status_code)
        print(response.text)
    return(text_index_i)

def find_diagnosis(data2_i):
    if pd.isna(data2_i['Corrected_Text_index']):
        diagnosis = ''
        icd10_code = ''
    else: 
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. For each given text, you need to output two outcomes: 1. The disease diagnosis based on the provided text, return with the format Diagnosis: diagnosis description; 2. The ICD-10-cm code of the diagnosis based on the output 1 and the given text context, return with the format ICD-10: the ICD-10-cm code."
                },
                {
                    "role": "user",
                    "content": f"Here are some examples:,{example_str}. Now, based on the above examples, please analyze the following text and provide the outcomes as described in the same format: "
                },
                {
                    "role": "user",
                    # Replace 'Your text here' with the actual text you want the model to analyze
                    "content": f"Your text here. {data2_i['Corrected_Text_index']}"
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        response_i = response.json()
        message = response_i['choices'][0]['message']['content']
        # print("Message: ", message)
        lines = message.split('\n')
        outputs = [line.split(':')[1].strip() for line in lines if ': ' in line]
        if len(outputs) == 2:
            diagnosis, icd10_code = outputs
        else:
            diagnosis = outputs
            icd10_code = outputs
            print("Unexpected format in assistant's response")
    return(diagnosis, icd10_code)

def process_batch(batch_org, file_num):
    """
    Processes a single batch of observations.

    Parameters:
    - batch: A list of observations.
    """
    batch = batch_org.copy()
    indexed_sens = []
    diagnoses = []
    icds = []
    # loop for each row in the batch
    for i, data_i in batch.iterrows():
        indexed_sen = index_sen(data_i)
        data_i['Corrected_Text_index'] = indexed_sen  # If data_i is a copy, this might not modify the original DataFrame
        diagnosis, icd10_code = find_diagnosis(data_i)
        indexed_sens.append(indexed_sen)
        diagnoses.append(diagnosis)
        icds.append(icd10_code)
    batch['Corrected_Text_index'] = indexed_sens
    batch['Diagnosis'] = diagnoses
    batch['ICD10'] = icds
    # Save the processed batch to a file
    output_filePath = f'/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1_{file_num}.csv'
    batch.to_csv(output_filePath, mode='a', header=False)

def batch_process(data, batch_size):
    """
    Divides the data into batches and processes each batch.
    Parameters:
    - data: The complete dataset.
    - batch_size: The number of observations per batch.
    """
    file_num = 0
    for i in range(0, len(data), batch_size):
        print(file_num)
        # skip when file_num is 0,1,2
        if file_num in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
            file_num += 1
            continue
        batch = data[i:i + batch_size]
        process_batch(batch,file_num)
        file_num += 1


#### run the batch process ####
batch_size = 100  
batch_process(sample_data, batch_size)


# import pandas as pd
# import os
# import numpy as np

# # read data from "/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1_0.csv"
# # add colnames to the data

# column_names = ['id','train_id','Text ID','Text','Sentences','Error Flag', 'Error Sentence ID','Error Sentence','Corrected Sentence','Corrected Text','Text_index','Diagnosis','ICD10']  # Define your column names here
# data = pd.read_csv("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1_0.csv", header=None, names=column_names)
# print(data.shape)

# for i in list(range(0, 22,1)):
#     if i == 0:
#         continue
#     data_temp = pd.read_csv("/Users/jiacong/Google Drive/Umich/EECS598_LLM/finalProject/results/TrainData_Step1_" + str(i) + ".csv", header=None, names=column_names)
#     data = pd.concat([data, data_temp], axis = 0)


