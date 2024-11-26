import os
import json
from pathlib import Path
import pickle
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from collections import Counter

from normalize_text import normalize

import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)



# if args.model_name == 'scibert':
model_name = 'm3rg-iitd/matscibert'
# elif args.model_name == 'bert':
#     model_name = 'bert-base-uncased'


model_revision = 'main'


max_seq_length = 1024
num_labels = 30000
config_kwargs = {
    'num_labels': num_labels,
    'revision': 'main',
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
    'model_max_length': max_seq_length
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)


def tokenize_function(examples):
    examples = normalize(examples)
    result = tokenizer(examples, truncation=True, padding='max_length', max_length=max_seq_length)
    return result

# Process each text file in the directory
def tokenize(directory, output_directory):
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            output_file_path = os.path.join(output_directory, filename.replace('.txt', '.json'))
            
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            tokenized_data = tokenize_function(text)
        #retxt = tokenizer.decode(tokenized_datasets['input_ids'])
            #print(tokenized_data)

            # Convert tensor data to lists for JSON serialization
            tokenized_data = {key: value for key, value in tokenized_data.items()}
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                #print(tokenized_data, len(tokenized_data['input_ids']))
                json.dump(tokenized_data, json_file)

    print("Tokenization complete. Files saved to:", output_directory)



def decode(directory):
     for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                token_data = json.load(file)# ['input_ids']
            #token_data = [30112, 30132, 496, 30138, 496, 4379, 205, 124, 145, 158, 546, 165, 23152, 147, 482, 116, 145, 158, 546, 422, 482, 116, 145, 170, 546, 422, 482, 116, 50000, 0, 102, 2329, 546, 422, 482, 116, 50000, 0, 102, 2329, 145, 4559, 30132, 546, 422, 286, 165, 6130, 3310, 18150, 7849, 390, 579, 2865, 7908, 137, 6592, 2632, 121, 111, 14512, 27119, 118, 30130, 1630, 583, 205, 111, 2329, 145, 158, 546, 165, 23152, 121, 106, 18799, 1461, 579, 10261, 50000, 0, 102, 5060, 145, 884, 30132, 50000, 0, 50000, 0, 102, 4193]
                         #[[30112, 30132, 496, 30138, 496, 4379, 205, 504, 145, 158, 546, 165, 23152, 147, 482, 116, 145, 158, 546, 422, 482, 116, 145, 170, 546, 422, 482, 116, 50000, 0, 102, 2329, 145, 239, 546, 239, 6687, 165, 1902, 4393, 585, 579, 2865, 7908, 137, 6592, 2632, 121, 111, 50000, 0, 102, 4193, 30138, 8280, 1630, 583, 205, 111, 1187, 165, 2799, 579, 8176, 137, 3685, 131, 502, 1529, 3428, 496, 30138, 145, 158, 546, 370, 165, 23152, 147, 874, 50000, 0, 102, 2329, 145, 158, 422, 50000, 0, 102, 4193, 30138, 50000]]
            token_data = [137, 6592, 2632, 121, 111, 13879, 5949, 579, 239, 30119, 1630, 583, 205, 6901, 145, 158, 546, 165, 23152, 147, 2781, 3552, 146, 145, 158, 546, 6448, 147, 592, 106, 4434, 131, 11881, 422, 3942, 422, 137, 3555, 579, 6404, 50000, 0, 50000, 0, 102, 1529, 255, 50000, 0, 102, 231, 30138, 246, 30146, 22235, 7452, 1879, 22755, 120, 1199, 50000, 0, 102, 2329, 30138, 165, 23152, 121, 145, 304, 579, 10011, 5840, 147, 592, 106, 20825, 320, 30132, 30116, 30138, 30116, 137, 6592, 2632, 121, 111, 3816, 23735, 3501]
            token_data = [50000, 102, 6901, 30111, 165, 2547, 585, 422, 8863, 7030, 7908, 137, 6592, 2632, 121, 111, 13879, 5949, 579, 239, 30119, 1630, 583, 205, 6901, 145, 158, 546, 165, 23152, 147, 2781, 3552, 146, 145, 158, 546, 6448, 147, 592, 106, 4434, 131, 11881, 137, 3555, 579, 6404, 6901, 30111, 30142, 26376, 25280, 205, 111, 11881, 579, 6404, 26376, 25280, 220, 302, 18212, 119, 205, 355, 6901, 145, 158, 546, 579, 146, 145, 158, 546, 3817, 8546, 220, 170, 205, 2532, 106, 205, 146, 145, 158, 546, 165, 23152, 147, 2781, 3552, 6901, 145, 158, 546, 6448, 147, 592, 106, 4434, 131, 11881, 137, 3555, 579, 6404, 6604, 30122, 30142, 26376, 25280, 205, 111, 11881, 579, 6404, 26376, 25280, 220, 302, 18212, 119, 205, 103, 0, 0, 0, 0, 0, 0, 0]
            detokenized_data =   tokenizer.decode(token_data, skip_special_tokens=True)
            print(detokenized_data)

            if_save = 0
            if if_save:
                txt_filename = filename.replace(".json", ".txt")
                txt_file_path = os.path.join(directory, txt_filename)
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(detokenized_data)
                print(f"Saved detokenized data to {txt_file_path}")





def count_tokens_in_directory(folder_path):
    token_counter = Counter()

    # Iterate through all JSON files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    tokens = data.get("input_ids", [])
                    token_counter.update(tokens)
                except json.JSONDecodeError:
                    print(f"Error reading file {filename}. Skipping...")

    print("Number of unique tokens:", len(token_counter))
    top_tokens_with_counts = token_counter.most_common(100)
    top_tokens = [token for token, count in top_tokens_with_counts]

    for token, count in top_tokens_with_counts:
       
        detokenized_data =   tokenizer.decode(token, skip_special_tokens=True)
        print(f"Token {token}: {count} : {detokenized_data}")



def main():
    # directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/cif2txt'
    #directory = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/temp'
    # output_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json_matscibert'
    # os.makedirs(output_directory, exist_ok=True)
    # tokenize(directory, output_directory)

    #count_tokens_in_directory(output_directory)

    # dec_directory = "/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json_matscibert"
    dec_directory = "/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/matsci_56_56_256_8"
    decode(dec_directory)


if __name__== '__main__':
    main()

