import os
import numpy as np
import json
import torch

# Paths to the folders
folder_B = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json/'
folder_A = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/dev/'
output_folder = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_json_npy'
os.makedirs(output_folder, exist_ok=True)

# Function to extract the mp number from the file name
def extract_mp_number(filename):
    # For .cif files
    if filename.endswith('.json'):
        return filename.split('.')[0].split('-')[1]
    # For .npy files
    elif filename.endswith('.npy'):
        return filename.split('_')[1].split('-')[1]
    return None

# Iterate through all files in folder B
for filename_b in os.listdir(folder_B):
    if filename_b.endswith('.json'):
        mp_number = extract_mp_number(filename_b)
        matching_files = []
        
        # Look for files in folder A that contain the same mp_number
        for filename_a in os.listdir(folder_A):
            if filename_a.endswith('.npy') and extract_mp_number(filename_a) == mp_number:
                matching_files.append(filename_a)
        
        # If matching files are found, read and combine the data
        if matching_files:
            filepath_b=f"{folder_B}{filename_b}"
            with open(filepath_b, 'r') as file:
                data = json.load(file)
                tokens = data['input_ids'][0]
                #tokens_tensor = torch.tensor(tokens, dtype=torch.float32)
            
            for file_a in matching_files:
                print(filename_b, file_a)
                filepath_a=f"{folder_A}{file_a}"
                npy_data = np.load(filepath_a)
                combined_data = (tokens, npy_data)

                # Save combined data to a new file
                output_filename = f'{output_folder}/combined{file_a[8:-4]}.pt'
                print(output_filename)
                torch.save( combined_data, output_filename)
        else: 
            print(f"No matching files found for: {filename_b}")


print("Processing complete. Combined files saved to:", output_folder)
