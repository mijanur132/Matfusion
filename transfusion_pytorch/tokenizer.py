import os
import json
from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  #matscibert

# Define the directory containing your text files
directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/cif'

# Ensure the output directory exists
output_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_text'
os.makedirs(output_directory, exist_ok=True)

# Process each text file in the directory
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        output_file_path = os.path.join(output_directory, filename.replace('.txt', '.json'))
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Tokenize the text
        tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors="pt")
        
        # Convert tensor data to lists for JSON serialization
        tokenized_data = {key: value.numpy().tolist() for key, value in tokenized.items()}
        
        # Save the tokenized data as JSON
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(tokenized_data, json_file)

print("Tokenization complete. Files saved to:", output_directory)
