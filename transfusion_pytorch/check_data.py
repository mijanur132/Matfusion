import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

# Path to the folder with .npy files
#folder_path = "/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/test"
folder_path = "/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/temp"

# Ensure the output directory for plots exists
output_folder = os.path.join(folder_path, "plots")
os.makedirs(output_folder, exist_ok=True)

def plot(path, cbed_stack):
    fig, axes = plt.subplots(1,3, figsize=(16,12))
    for ax, cbed in zip(axes.flatten(), cbed_stack):
        ax.imshow(cbed, cmap='gnuplot2')
        ax.axis('off')
    path=f"{path}.png"
    print(path)
    plt.savefig(path)

# Loop over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):  # Process only .npy files
        file_path = os.path.join(folder_path, filename)
        # Load the numpy file
        data = np.load(file_path)
        print(f"data: {data[data>10], len(data[data>10])}")
        # Calculate min and max
        data_min = data.min()
        data_max = data.max()
        data_s = (data-data.min())/(data.max()-data.min())
        print(file_path, data_min, data_max, data_s.max())
        output_file_path = os.path.join(output_folder, filename.replace(".npy", ".png"))
        plot(output_file_path, data_s)

        # print(data_s[0], data_s[0].shape, data_s[0].max())
        # fig=im.fromarray(np.uint8((25500*data_s[0])))
        # op_path = f"{output_file_path}_pil.png"
        # print(f"saved:{op_path}")
        # fig.save(op_path)
     