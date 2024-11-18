from transformers import BertTokenizer
import torch
import os
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def plot(path, cbed_stack):
    fig, axes = plt.subplots(1,3, figsize=(16,12))
    for ax, cbed in zip(axes.flatten(), cbed_stack):
        ax.imshow(cbed, cmap='gnuplot2')
        ax.axis('off')
    path=f"{path}.png"
    plt.savefig(path)

def plot_1d(path, cbed_data):
    plt.figure(figsize=(16, 12))  # Adjust figure size as needed
    plt.imshow(cbed_data, cmap='gnuplot2')
    plt.axis('off')  # Turn off axis for a cleaner look
    
    # Save the plot as a PNG file with the given path
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def text_process():
    file = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/mod_128_sample_out_10.pt'

    input = torch.load(file)
    tokens = input[0][1:]
    print("tokens:", tokens)
    decoded_text = tokenizer.decode(tokens)
    out_file = f'{file}.txt'
    print(decoded_text)
    with open(out_file, 'w') as f:
        f.write(decoded_text)

def mod_process():
    path = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/mod_128_sample_out_16.pt'
    print(f"input:{path}")
    input = torch.load(path)
    print(input)
    tokens = input[2][1:][0]
    print(tokens)
    n=np.array(tokens.cpu())
    print(f"array:{n},{n.shape}")
    ns = (n-n.min())/(n.max()-n.min())
    print(n.min(), n.max())
    fig=im.fromarray(np.uint8((ns*255)))
    op_path = f"{path}.png"
    print(f"saved:{op_path}")
    fig.save(op_path)
    op_path = f"{path}_1.png"
    plot_1d(op_path, ns)

def mod_process2():
    path= "/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/pred_flow_20241114-174003.npy"
   # path= "/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/pred_flow.npy"
    z=np.load(path)
    fig=im.fromarray(np.uint8(z[0]*255))
    op_path = f"{path}.png"
    fig.save(op_path)

if __name__ == "__main__":
    #mod_process()
    mod_process2()
    #text_process()

   
