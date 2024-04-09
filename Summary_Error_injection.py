"""
Author: Yufeng Zhanbg
Date: April 6th 2024
"""

import numpy as np
import pandas as pd
import os,sys
import re
import pickle
import random


class summary_generator():
    def __init__(self,
                 dir_root,
                 disease_file_root,
                 discharge_file = 'discharge.csv',
                 disease_file = 'discharge_infect_dull.csv',
                 shuffle_ratio = 0.6,
                 show_partial = False, 
                 num = 10):
        
        self.disease = pd.read_csv(os.path.join(disease_file_root,disease_file),index_col = 0).drop(['text'],axis = 1)
        if show_partial:
            self.discharge_raw = pd.read_csv(os.path.join(dir_root, discharge_file), nrows=num)
        else: 
            self.discharge_raw = pd.read_csv(os.path.join(dir_root,discharge_file))
        self.discharge = pd.merge(self.disease, self.discharge_raw,on = ['note_id', 'subject_id', 'hadm_id'])
        self.discharge['gpt_icd_10'] = self.discharge['gpt_icd_10'].apply(lambda x: x.strip())
        
        ICD_phecode = pd.read_csv(os.path.join('./data_temp','Phecode_map_icd10cm_beta.csv'),encoding = "ISO-8859-1")
        ICD_phecode['phecode'] = ICD_phecode['phecode'].astype(str)
        self.ICD_string = ICD_phecode[['icd10cm','icd10cm_str']].copy()
        self.ICD_string.columns = ['gpt_icd_10','gpt_icd_str']
        self.ICD_string['gpt_icd_10'] = self.ICD_string['gpt_icd_10'].astype(str)
        self.discharge = pd.merge(self.discharge,self.ICD_string,on = 'gpt_icd_10')


        self.discharge['gpt_icd_corrputed_root'] = self.discharge['gpt_icd_10'].apply(self.composite_corrput)
        self.discharge[['gpt_icd_corrputed', 'corrputed_str']] = self.discharge['gpt_icd_corrputed_root'].apply(self.extract_str_meaning).tolist()
        # self.discharge = self.discharge[self.discharge.corrputed_str.notna()].reset_index()



        
        
        self._initialize_param()
        self.insert_positions = {}
        
        # complaint
        self.chief_complaint_pattern1 = re.compile(r"Chief Complaint:\n(.+?)\n")
        self.chief_complaint_pattern2 = re.compile(r" Complaint:\n(.+?)\n")
        # diagnosis
        self.diagnosis_pattern1 = re.compile(r"Discharge Diagnosis:\n(.*?)\n\s+\nDischarge Condition:", re.DOTALL)
        self.primary_diagnosis_pattern1 = r'Primary:\s*([^,]+)'
        self.primary_diagnosis_pattern2 = r'PRIMARY DIAGNOSIS[:.]\s*([^,\.]+)'
    
        # vital signs
        self.vital_sign_pattern1 = re.compile(r"\nVS(?![a-zA-Z])\s*(.+?)\n")
        self.vital_sign_pattern2 = re.compile(r"\nVitals\s*(.+?)\n")
        self.vital_sign_pattern3 = re.compile(r"\nPhysical Exam:\nO:(.+?)\n")
        self.vital_sign_pattern4 = re.compile(r"\nPHYSICAL EXAM ON ADMISSION:\nO:(.+?)\n")
    
        # history illness present
        self.history_pattern_1 = re.compile(r"History of Present Illness:\s*(.*?)(?=\n\s+\nPast Medical History:)", re.DOTALL)
        self.history_pattern_2 = re.compile(r"History of Present Illness:\s*(.*?)(?=\nREVIEW OF SYSTEMS:)", re.DOTALL)
    
        # Gender
        self.gender_pattern = re.compile(r'Sex:\s*(\w)\n')

        self.shuffle_ratio = shuffle_ratio

    
    
    def _initialize_param(self):
        self.n_summaries = 0
        self.indices = None
        self.indices_to_modify = None
    
    
    def _condense_text(self,text):
        text = re.sub(r'\=+', '', text)
        text = re.sub(r'\n\n', '. ', text) 
        text = re.sub(r'(?<!:)\n', ', ', text) 
        text = re.sub(r'\n', ' ', text) 
        text = re.sub(r'\-|_', '', text) 
        return text.strip()

    def _condense_illness_text(self,text):
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'\n+', ' ', text)  
        return text.strip()

    def _customize_BP(self,string):
        bp_pattern = re.compile(r'\d+\s*/\s*\d+')
        bp_readings = bp_pattern.findall(string)
        for bp in bp_readings:
            string = string.replace(bp, "BP_PLACEHOLDER")
        splits = string.split()
        for i, s in enumerate(splits):
            if 'BP_PLACEHOLDER' in s:
                splits[i] = bp_readings.pop(0)
        return splits

    def gen_complaint(self, note):
        if re.search(self.chief_complaint_pattern1, note):
            matched = re.search(self.chief_complaint_pattern1, note).group(1)
        elif re.search(self.chief_complaint_pattern2, note):
            matched = re.search(self.chief_complaint_pattern2, note).group(1)
        else:
            matched = None
    
        if matched:
            text = matched.strip()
            return text
        else:
            return np.nan
    
    def gen_diagnosis(self, note):
        matched = re.search(self.diagnosis_pattern1, note)
        if matched:
            text = matched.group(1).strip()
            text = self._condense_text(text)
    
            primary_diagnosis = None
            if re.search(self.primary_diagnosis_pattern1, text, re.IGNORECASE):
                primary_diagnosis = re.search(self.primary_diagnosis_pattern1, text, re.IGNORECASE)
            elif re.search(self.primary_diagnosis_pattern2, text, re.IGNORECASE):
                primary_diagnosis = re.search(self.primary_diagnosis_pattern2, text, re.IGNORECASE)
    
            primary_diagnosis_match = primary_diagnosis.group(1) if primary_diagnosis else text
            return primary_diagnosis_match
        else:
            return np.nan


    def gen_vitals(self,note):
        if re.search(self.vital_sign_pattern1, note):
            matched = re.search(self.vital_sign_pattern1, note).group(1)
            splits = 1
        elif re.search(self.vital_sign_pattern2, note):
            matched = re.search(self.vital_sign_pattern2, note).group(1)
            splits = 2
        elif re.search(self.vital_sign_pattern3, note):
            matched = re.search(self.vital_sign_pattern3, note).group(1)
            splits = 3
        elif re.search(self.vital_sign_pattern4, note):
            matched = re.search(self.vital_sign_pattern4, note).group(1)
            splits = 4
        else:
            splits = None
        if splits:
            splits = self._customize_BP(matched)
            splits = [s for s in splits if 'O2' not in s ]
            splits = [s for s in splits if re.search(r'\d+|\d+/\d+', s)]
            splits = [re.sub(r'[a-zA-Z]', '', s) for s in splits]
            pattern = re.compile(r'[^\w/\.]|_')
            splits = [pattern.sub('', s) for s in splits]
            if len(splits) >= 5: 
                return f'The Temperature is {splits[0]}, BP is {splits[1]}, HR is {splits[2]}, RR is {splits[3]}, SpO2 is {splits[4]}.'
            elif len(splits) == 4: 
                return f'The Temperature is {splits[0]}, BP is {splits[1]}, HR is {splits[2]}, RR is {splits[3]}.'
            elif len(splits) == 3: 
                return f'The Temperature is {splits[0]}, BP is {splits[1]}, HR is {splits[2]}.'
            else:
                return np.nan
        else:
            return np.nan

    def gen_history_present_illness(self,note):
    
        if re.search(self.history_pattern_2, note):
            matched = re.search(self.history_pattern_2, note).group(1)
        elif re.search(self.history_pattern_1, note):
            matched = re.search(self.history_pattern_1, note).group(1)
        else:
            matched = None
        if matched:
            text = matched.strip()
            text = self._condense_illness_text(text)
            return text
        else:
            return np.nan

    def gen_gender(self,note):
        match = self.gender_pattern.search(note)
        if match:
            g = match.group(1).strip()
            if g == 'F':
                return 'She'
            elif g == 'M':
                return 'He'
        else:
            return np.nan
    
    def gen_corrupted_diagnosis(self,idx):
        return self.discharge.loc[idx,'corrputed_str'].strip()

    def _order_sentences(self,sentences):
        return [f"{idx}. {sentence}" for idx, sentence in enumerate(sentences) ]

    def _unorder_sentences(self,senteces):
        return re.sub(r"^\d+\.\s*", "", senteces)

    def extract_icd_root(self,x):
        if '.' in x:
            return x.split('.')[0].strip()
        else:
            return x

    def generate_adjacent_icd_codes(self,root_code):
        if not root_code or len(root_code) < 2:
            return None
    
        adjacent_codes = []
        base, last_char = root_code[:-1], root_code[-1]
    
        # Expand the search scope for digits
        if last_char.isdigit():
            digit = int(last_char)
            adjacent_digits = [str(i) for i in range(max(0, digit - 2), min(9, digit + 3))]
            adjacent_codes.extend([base + d for d in adjacent_digits if d != last_char])
    
        # Expand the search scope for letters
        elif last_char.isalpha():
            adjacent_letters = [chr(i) for i in range(max(ord('A'), ord(last_char) - 2), min(ord('Z'), ord(last_char) + 3))]
            adjacent_codes.extend([base + l for l in adjacent_letters if l != last_char])
    
        # Randomly choose one of the adjacent codes if available
        return random.choice(adjacent_codes) if adjacent_codes else None


    def composite_corrput(self,icd_code):
        root_code = self.extract_icd_root(icd_code)
        adjacent_code = self.generate_adjacent_icd_codes(root_code)
        return adjacent_code

    def extract_str_meaning(self,x):
        x = x.strip()
        matches = self.ICD_string[self.ICD_string['gpt_icd_10'].str.contains(x)]
        if len(matches) > 1:
            random_matches = matches.sample()
            c, s = random_matches.iloc[0][['gpt_icd_10','gpt_icd_str']]
            return [c, s]
        else:
            return [np.nan, np.nan]

    def custom_tokenizer(self, text, delimiter='. '):

        potential_sentences = text.split('. ')
        sentences = [s for s in potential_sentences if not s.startswith('___') and not s.endswith('___')]
        sentences = [s for s in potential_sentences if not s.lower().startswith('secondary')]
        sentences = [sentence for sentence in sentences if '___' not in sentence and 'Mr' not in sentence]
        sentences = [sentence for sentence in sentences if sentence.strip()]
        filtered_sentences = self._order_sentences(sentences)
        return filtered_sentences

    def random_inject_sentences(self,text):
        sentences = text.split('\n')
        last_sentence = sentences[-1]
        rest_sentences = sentences[:-1]
        insert_position = random.randint(2, len(sentences)-1)
        rest_sentences.insert(insert_position, last_sentence)
        rearranged_text = [self._unorder_sentences(s) for s in rest_sentences]
        rearranged_text = self._order_sentences(rearranged_text)
        return "\n".join(rearranged_text),insert_position,last_sentence

    def inject_senteces_by_index(self,text,insert_position):
        sentences = text.split('\n')
        last_sentence = sentences[-1]
        rest_sentences = sentences[:-1]
        rest_sentences.insert(insert_position, last_sentence)
        rearranged_text = [self._unorder_sentences(s) for s in rest_sentences]
        rearranged_text = self._order_sentences(rearranged_text)
        return "\n".join(rearranged_text),insert_position,last_sentence
    
    
    def generate_correct_summary(self):
        structured_summaries = []
        for idx, row in self.discharge.iterrows():
            summary = []
            gender = self.gen_gender(row['text'])
            complaint = self.gen_complaint(row['text'])
            diagnosis = self.gen_diagnosis(row['text'])
            vitals = self.gen_vitals(row['text'])
            illness = self.gen_history_present_illness(row['text'])

            if gender:
                pronoun = gender  # 'He' or 'She'
                if complaint:
                    summary.append(f"{pronoun} presented with a chief complaint of {complaint}.")
                if illness:
                    summary.append(f"{pronoun} has a history of {illness}.")
                if vitals:
                    summary.append(f"Upon examination, {pronoun}'s vital signs were: {vitals}.")
                if diagnosis:
                    summary.append(f"The primary diagnosis was determined to be {diagnosis}.")

            structured_summary = self.custom_tokenizer(" ".join(summary))
            if len(structured_summary) <= 4:
                structured_summary = 'ISINF' # not enought lengths
            else:
                structured_summary = "\n".join(structured_summary)
                self.n_summaries += 1 # meaningful summaries
            structured_summaries.append(structured_summary)
        self.discharge['correct_summary'] = structured_summaries


    def generate_corrupted_summary(self):
        structured_summaries = []
        for idx, row in self.discharge.iterrows():
            summary = []
            gender = self.gen_gender(row['text'])
            complaint = self.gen_complaint(row['text'])
            diagnosis = self.gen_corrupted_diagnosis(idx)
            vitals = self.gen_vitals(row['text'])
            illness = self.gen_history_present_illness(row['text'])

            if gender:
                pronoun = gender  # 'He' or 'She'
                if complaint:
                    summary.append(f"{pronoun} presented with a chief complaint of {complaint}.")
                if illness:
                    summary.append(f"{pronoun} has a history of {illness}.")
                if vitals:
                    summary.append(f"Upon examination, {pronoun}'s vital signs were: {vitals}.")
                if diagnosis:
                    summary.append(f"The primary diagnosis was determined to be {diagnosis}.")

            structured_summary = self.custom_tokenizer(" ".join(summary))
            if len(structured_summary) <= 4:
                structured_summary = 'ISINF' # not enought lengths
            else:
                structured_summary = "\n".join(structured_summary)
            structured_summaries.append(structured_summary)
        self.discharge['corrupted_summary'] = structured_summaries
    
    
    def records_len_incons(self):
        self.inconsis_lineidx = []
        for idx, row in self.discharge.iterrows():
            number_of_correct = row['correct_summary'].count('\n') + 1  # Adding 1 for the last line
            number_of_corrupted = row['corrupted_summary'].count('\n') + 1 
            if number_of_correct != number_of_corrupted:
                self.inconsis_lineidx.append(idx)
        
    
    
    def random_injection(self):
        number_to_modify = int(self.n_summaries * self.shuffle_ratio) # n_summaries: valid summries
        self.indices = [i for i, text in enumerate(self.discharge['correct_summary']) if text != 'ISINF'] # self.indices: valid summaries index
        self.indices_to_modify = random.sample(self.indices, number_to_modify) # valid summaries modifibale index
        
        correct_texts_to_modify = [self.discharge.loc[i,'correct_summary'] for i in self.indices_to_modify]
        corrupted_texts_to_modify = [self.discharge.loc[i,'corrupted_summary'] for i in self.indices_to_modify]    
        
        print(f'Within {str(self.n_summaries)} summaries, {str(len(self.indices))} need to be modified.')
            

        diag_indices = []
        
        for i,text in enumerate(self.discharge['correct_summary']):
            if i in self.indices_to_modify:
                modified_text, position, sentence = self.random_inject_sentences(text)
                self.discharge.loc[i,'correct_summary'] = modified_text
                self.discharge.loc[i,'correct_sentence'] = self._unorder_sentences(sentence)
                self.insert_positions[i] = position
                diag_indices.append(position)
            else:
                text = self.discharge.loc[i,'correct_summary']
                diag_indices.append(text.count('\n') )
                last_sentence = text.split('\n')[-1]
                self.discharge.loc[i,'correct_sentence'] = last_sentence
        
                                    
        for i,text in enumerate(self.discharge['corrupted_summary']):
            if i in self.indices_to_modify:
                modified_text, _, sentence = self.inject_senteces_by_index(text,self.insert_positions[i])
                self.discharge.loc[i,'corrupted_summary'] = modified_text
                self.discharge.loc[i,'corrupted_sentence'] = self._unorder_sentences(sentence)
            else:
                text = self.discharge.loc[i,'corrupted_summary']
                last_sentence = text.split('\n')[-1]
                self.discharge.loc[i,'corrupted_sentence'] = last_sentence
                

        self.discharge['sentence_id'] = diag_indices
        self.records_len_incons()


    def filter_out_inconsistent_records(self):
        mask = ~self.discharge.index.isin(self.inconsis_lineidx)
        self.discharge = self.discharge[mask]
disease_name = 'circulatory_system'

MIMIC_summary_generator = summary_generator(dir_root,
                                            disease_file_root,
                                            discharge_file = 'discharge.csv',
                                            disease_file = f'discharge_{disease_name}_final.csv',shuffle_ratio = 0.6)
MIMIC_summary_generator.generate_correct_summary()
MIMIC_summary_generator.generate_corrupted_summary() 
MIMIC_summary_generator.random_injection()
print(len(MIMIC_summary_generator.inconsis_lineidx))
MIMIC_summary_generator.filter_out_inconsistent_records()
MIMIC_summary_generator.records_len_incons()
print(len(MIMIC_summary_generator.inconsis_lineidx))
df = MIMIC_summary_generator.discharge
