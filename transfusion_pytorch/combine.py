import os
import numpy as np
import json
import torch

# Paths to the folders
folder_A = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json_matscibert/'
folder_B = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/combined/'
output_folder = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_json_npy_matscibert'
os.makedirs(output_folder, exist_ok=True)

# Obtain SLURM environment variables
task_id = int(os.getenv('SLURM_PROCID', 0))
num_tasks = int(os.getenv('SLURM_NTASKS', 1))

# Function to extract the mp number from the file name
def extract_mp_number(filename):
    #print(filename)
    if filename.endswith('.json'):
        a=filename.split('.')[0].split('-')[1]
       #print(a)
        return a
    elif filename.endswith('.npy'):
        b=filename.split('_')[1].split('-')[1]
        #print(b)
        return b
    return None

a_files_dict = {}  #jsons..
for filename_a in os.listdir(folder_A):
    mp_number = extract_mp_number(filename_a)
    a_files_dict[mp_number] = filename_a
print("dict 1 done..", len(a_files_dict))
b_files_dict = {}  #numpys
for filename_b in os.listdir(folder_B):
    mp_number = extract_mp_number(filename_b)
    b_files_dict.setdefault(mp_number, []).append(filename_b)

total_length = sum(len(lst) for lst in b_files_dict.values())
print("dict 2 done..", total_length)
x=0
y=0
z=0

for i, mp_number in enumerate(a_files_dict.keys()): #json
    # if i % num_tasks != task_id:
    #     continue
# for mp_number in a_files_dict:  #for each json
    x+=1
    print("total_file A:",x)
    if mp_number in b_files_dict:  
        y+=1
        filename_a = a_files_dict[mp_number]
        filepath_a = os.path.join(folder_A, filename_a)
        print("fileA:",filepath_a)
        with open(filepath_a, 'r') as file:
            data = json.load(file)
            tokens = data['input_ids'][0]
        print("fileA loaded")
        for filename_b in b_files_dict[mp_number]:
            print("filenameb:", filename_b)
            filepath_b = os.path.join(folder_B, filename_b)
            npy_data = np.load(filepath_b)
            combined_data = (tokens, npy_data)
            output_filename = os.path.join(output_folder, f'combined{filename_b[8:-4]}.pt')
            print(f"saved{output_filename}  {y}")
            torch.save(combined_data, output_filename)
    else:
        z+=1
        print("no match:", z)
