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

            detokenized_data =   tokenizer.decode(token_data, skip_special_tokens=True)
            print(detokenized_data)




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
    dec_directory = "/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/matsci_28_28_256_4"
    decode(dec_directory)


if __name__== '__main__':
    main()

