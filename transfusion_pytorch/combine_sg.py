import os
import numpy as np
import json
import torch
import re

# Paths to the folders
folder_A = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json_matscibert/'
folder_B = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/combined/'
output_folder = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_sg_npy'
os.makedirs(output_folder, exist_ok=True)

# Obtain SLURM environment variables
task_id = int(os.getenv('SLURM_PROCID', 0))
num_tasks = int(os.getenv('SLURM_NTASKS', 1))

# Function to extract the mp number from the file name

def extract_sg(filename):
    match = re.search(r'sg_(\d+)\.npy', filename)
    print(int(match.group(1)))
    if match:
        # Extract the number and convert it to an integer
        return int(match.group(1))
    else:
        # Return None if no matching pattern is found
        return None

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


b_files_dict = {}  #numpys
sg_dict = {}
for filename_b in os.listdir(folder_B):
    sg = extract_sg(filename_b)
    filepath_b = os.path.join(folder_B, filename_b)
    npy_data = np.load(filepath_b)
    print(filename_b,sg)
    combined_data = (sg, npy_data)
    output_filename = os.path.join(output_folder, f'combined{filename_b[8:-4]}.pt')
    print(f"saved{output_filename}")
    torch.save(combined_data, output_filename)
