import os
from openai import OpenAI
import pandas as pd
import re
import numpy as np

api_key = "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"

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

def filter_content(rec):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are an biomedical AI in analyzing eletronic health records."},
            {
                "role": "user",
                "content": """i have a UW clinical record dataset, which contains clinical reports as follow:

    Example 1:
    0 A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.
    1 He works as a commercial fisherman on Lake Superior.
    2 Current medications include metoprolol and warfarin.
    3 His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm
    4 Hg.
    5 Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.
    6 After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae.
    7 An x-ray of the chest showed consolidation of the right upper lobe.

    Example 2:
    0 A 9-year-old girl is brought to the pediatrician by her mother who reports that the girl has been complaining of genital itching over the past few days.
    1 She states she has noticed her daughter scratching her buttocks and anus for the past week; however, now she is scratching her groin quite profusely as well.
    2 The mother notices that symptoms seem to be worse at night.
    3 The girl is otherwise healthy, is up to date on her vaccinations, and feels well.
    4 She was recently treated with amoxicillin for a middle ear infection.
    5 The child also had a recent bought of diarrhea that was profuse and watery that seems to be improving.
    6 Her temperature is 98.5 F (36.9 C), blood pressure is 111/70
    7 mmHg, pulse is
    8 83/min, respirations are 16/min, and oxygen saturation is 98% on room air.
    9 Physical exam is notable for excoriations over the girl's anus and near her vagina.
    10 Suspected of infection with Giardia lamblia.

    Example 3:
    0 Blood cultures are sent to the laboratory.
    1 Intravenous antibiotic therapy is started.
    2 Transesophageal echocardiography shows a large, oscillating vegetation attached to the tricuspid valve.
    3 Causal organism is Staphylococcus epidermidis.
    4 There are multiple small vegetations attached to tips of the tricuspid valve leaflets.
    5 There is moderate tricuspid regurgitation.
    6 The left side of the heart and the ejection fraction are normal."""},
        {
                "role": "user", 
                "content": f"""I'm converting clinical records from the MIMIC database to resemble records from the UW dataset. 
For each sentence, decide whether it contains information about clinical conditions, diagnostic evaluations, or treatment plans. If so, keep the sentence. Otherwise, drop the sentence.
Return the rewritten sentences directly without sentence indexes or any other prefixes. One sentence a line.
Record:
{rec}"""},
                ],
        model="gpt-3.5-turbo-0125",
        temperature = 0
    )
    return chat_completion.choices[0].message.content


def modify_term(rec):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are an biomedical AI in analyzing eletronic health records."},
            {
                "role": "user",
                "content": """I have a UW clinical record dataset, which contains clinical reports as follow:

    Example 1:
    0 A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.
    1 He works as a commercial fisherman on Lake Superior.
    2 Current medications include metoprolol and warfarin.
    3 His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm
    4 Hg.
    5 Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.
    6 After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae.
    7 An x-ray of the chest showed consolidation of the right upper lobe.

    Example 2:
    0 A 9-year-old girl is brought to the pediatrician by her mother who reports that the girl has been complaining of genital itching over the past few days.
    1 She states she has noticed her daughter scratching her buttocks and anus for the past week; however, now she is scratching her groin quite profusely as well.
    2 The mother notices that symptoms seem to be worse at night.
    3 The girl is otherwise healthy, is up to date on her vaccinations, and feels well.
    4 She was recently treated with amoxicillin for a middle ear infection.
    5 The child also had a recent bought of diarrhea that was profuse and watery that seems to be improving.
    6 Her temperature is 98.5 F (36.9 C), blood pressure is 111/70
    7 mmHg, pulse is
    8 83/min, respirations are 16/min, and oxygen saturation is 98% on room air.
    9 Physical exam is notable for excoriations over the girl's anus and near her vagina.
    10 Suspected of infection with Giardia lamblia.

    Example 3:
    0 Blood cultures are sent to the laboratory.
    1 Intravenous antibiotic therapy is started.
    2 Transesophageal echocardiography shows a large, oscillating vegetation attached to the tricuspid valve.
    3 Causal organism is Staphylococcus epidermidis.
    4 There are multiple small vegetations attached to tips of the tricuspid valve leaflets.
    5 There is moderate tricuspid regurgitation.
    6 The left side of the heart and the ejection fraction are normal."""},
        {
                "role": "user", 
                "content": f"""I'm converting clinical records from the MIMIC database to resemble records from the UW dataset. Especially, replace the diagnoses terminology in the input record with layman's terminology if the terminology is longer than 4 words, and avoid providing detailed explanation of the medical concepts (e.g., the subtype of a terminology). Constraints: Keep the sentence indexes UNCHANGED. Return the rewritten record only and directly. One sentence a line.
    Record:
    {rec}"""},
                ],
        model="gpt-3.5-turbo-0125",
        temperature = 0
    )
    return chat_completion.choices[0].message.content

def modify_term_target(rec, term):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are an biomedical AI in analyzing eletronic health records."},
        {
                "role": "user", 
                "content": f"""If the terminology "{term}" is longer than 4 tokens: replace it in the input with layman's terminology, removing providing detailed explanation of the medical concepts (e.g., the subtype of a terminology). Otherwise: keep it as it is, and do not rewrite the sentence. Return the rewritten record only and directly.
    Input:
    {rec}"""},
                ],
        model="gpt-3.5-turbo-0125",
        temperature = 0
    )
    return chat_completion.choices[0].message.content

def enhance_format(rec):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are an biomedical AI in analyzing eletronic health records."},
            {
                "role": "user",
                "content": """I have a UW clinical record dataset, which contains clinical reports as follow:

    Example 1:
    0 A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.
    1 He works as a commercial fisherman on Lake Superior.
    2 Current medications include metoprolol and warfarin.
    3 His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm
    4 Hg.
    5 Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.
    6 After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae.
    7 An x-ray of the chest showed consolidation of the right upper lobe.

    Example 2:
    0 A 9-year-old girl is brought to the pediatrician by her mother who reports that the girl has been complaining of genital itching over the past few days.
    1 She states she has noticed her daughter scratching her buttocks and anus for the past week; however, now she is scratching her groin quite profusely as well.
    2 The mother notices that symptoms seem to be worse at night.
    3 The girl is otherwise healthy, is up to date on her vaccinations, and feels well.
    4 She was recently treated with amoxicillin for a middle ear infection.
    5 The child also had a recent bought of diarrhea that was profuse and watery that seems to be improving.
    6 Her temperature is 98.5 F (36.9 C), blood pressure is 111/70
    7 mmHg, pulse is
    8 83/min, respirations are 16/min, and oxygen saturation is 98% on room air.
    9 Physical exam is notable for excoriations over the girl's anus and near her vagina.
    10 Suspected of infection with Giardia lamblia.

    Example 3:
    0 Blood cultures are sent to the laboratory.
    1 Intravenous antibiotic therapy is started.
    2 Transesophageal echocardiography shows a large, oscillating vegetation attached to the tricuspid valve.
    3 Causal organism is Staphylococcus epidermidis.
    4 There are multiple small vegetations attached to tips of the tricuspid valve leaflets.
    5 There is moderate tricuspid regurgitation.
    6 The left side of the heart and the ejection fraction are normal."""},
        {
                "role": "user", 
                "content": f"""I'm converting clinical records from the MIMIC database to resemble records from the UW dataset. For each sentence in the following record, polish its writing style, structure, and format based on the provided examples in the UW dataset, which is concise but easy to understand. Constraints: Keep the sentence indexes UNCHANGED. Return the rewritten sentences directly without sentence indexes or any other prefixes. Return the rewritten record only and directly. One sentence a line.
    Record:
    {rec}"""},
                ],
        model="gpt-3.5-turbo-0125",
        temperature = 0
    )
    return chat_completion.choices[0].message.content

def enhance_mimic(row):
    try:
        corrputed_str = row['corrputed_str']
        correct_str = row['primary_diagnosis']
        nid = row['sentence_id']
        text = row['corrupted_summary']
        recs = text.split('\n')
        m = re.search(r'^[0-9]*\.*? *([a-zA-Z].+)', row['corrupted_sentence'])
        if m is not None:
            s1 = m.group(1)
        else:
            s1 = row['corrupted_sentence']
        m = re.search(r'^[0-9]*\.*? *([a-zA-Z].+)', row['correct_sentence'])
        if m is not None:
            s2 = m.group(1)
        else:
            s2 = row['correct_sentence']

        rec = '\n'.join(recs[:nid] + recs[nid+1:])
        rec_1 = filter_content(rec)
        rec_2 = modify_term(rec_1)
        rec_3 = enhance_format(rec_2)
        if ',' in corrputed_str:
            d1 = modify_term_target(s1, corrputed_str)
        else:
            d1 = s1
        if ',' in correct_str:
            d2 = modify_term_target(s2, correct_str)
        else:
            d2 = s2
        lst = rec_3.split('\n')
        lst = [re.sub('^[^a-zA-Z0-9]', '', l).strip() for l in lst] # remove prefix if any
        nid = np.random.choice(range(len(lst)), p=[0.6/(len(lst)-1)]*(len(lst)-1)+[0.4])
        if len(lst) > nid:
            rec_4 = lst[:nid] + [d2] + lst[nid+1:]
            rec_5 = lst[:nid] + [d1] + lst[nid+1:]
        else:
            rec_4 = lst + [d2]
            rec_5 = lst + [d1]
        corrupted_rec = '\n'.join([f"{i}. {sent}" for i, sent in enumerate(rec_5)])
        correct_rec = '\n'.join([f"{i}. {sent}" for i, sent in enumerate(rec_4)])
        return correct_rec, corrupted_rec, d2, d1, nid
    except:
        return None, None, None, None, None

mimic = pd.read_csv('/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/corrupt_files/merged_corrupted_mimic.shuffle.csv').iloc[1000:2000]
mimic_sub = mimic[['correct_summary', 'corrupted_summary', 'sentence_id', 'corrputed_str', 'primary_diagnosis', 'correct_sentence', 'corrupted_sentence']]
res = mimic_sub.apply(enhance_mimic, axis=1, result_type ='expand')
res.columns = ["correct_summary", 'corrupted_summary', 'correct_sentence', 'corrupted_sentence', 'sentence_id']
res.to_csv('/nfs/turbo/umms-drjieliu/usr/hyhao/eecs598/data/corrupt_files/merged_corrupted_mimic_enhanced.n1000_2.csv', index=False)