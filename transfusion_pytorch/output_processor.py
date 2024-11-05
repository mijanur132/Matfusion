from transformers import BertTokenizer
import torch
import os
import numpy

path = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#for item in os.listdir(path):
item = 'sample_out_20.pt'
file = f'{path}/{item}'
print("opened:", file)
input = torch.load(file)
tokens = input[0][1:]
print("tokens:", tokens)
decoded_text = tokenizer.decode(tokens)
out_file = f'{file}.txt'
print(decoded_text)
with open(out_file, 'w') as f:
    f.write(decoded_text)

