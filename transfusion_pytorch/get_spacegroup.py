import os

# Define the path to the main directory (replace 'path/to/main_directory' with your actual path)
main_directory = '/lustre/orion/stf218/proj-shared/junqi/materialsgenomics'

# Output file to store the results
output_filename = 'mp2space_group.txt'
op_2 = "top100sg.txt"


# Dictionary to store subfolder prefixes and file names
subfolder_files = {}

# Traverse through each item in the main directory
for dir_entry in os.listdir(main_directory):
    dir_path = os.path.join(main_directory, dir_entry)
    # Check if it's a directory
    if os.path.isdir(dir_path):
        # Get the part of the subfolder name before the first "_"
        subfolder_prefix = dir_entry.split('_')[0]
        # Convert subfolder_prefix to integer if possible
        try:
            subfolder_prefix = int(subfolder_prefix)
        except ValueError:
            continue  # Skip the directory if prefix isn't a valid integer

        # Initialize the list for storing file names if new
        if subfolder_prefix not in subfolder_files:
            subfolder_files[subfolder_prefix] = []

        # Traverse through each file in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith('.cif'):
                # Append the file name without the '.cif' extension
                subfolder_files[subfolder_prefix].append(filename[:-4])

# Open the output file in write mode
with open(output_filename, 'w') as file:
    # Sort the dictionary by keys (subfolder_prefix) in descending order
    for key in sorted(subfolder_files, reverse=False):
        # Write each file name with the corresponding subfolder prefix
        for filename in subfolder_files[key]:
            print(filename, key)
            file.write(f"{filename}:{key}\n")

# Determine top 100 subfolder prefixes by the number of files, sorted by file count in descending order
top_100_subfolders = sorted(subfolder_files.items(), key=lambda x: len(x[1]), reverse=True)[:100]

# Save the top 100 subfolder prefixes and their file counts to a separate file
with open(op_2, 'w') as file:
    for prefix, files in top_100_subfolders:
        print(f"Prefix {prefix}: {len(files)} files\n")
        file.write(f"Prefix {prefix}: {len(files)} files\n")

# Print completion message
print(f"Data has been saved to {output_filename}")


