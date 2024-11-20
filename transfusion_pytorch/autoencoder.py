import json
import sys
import os
import glob
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from torch import nn, tensor, randint, randn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchinfo import summary

from einops import rearrange
from einops.layers.torch import Rearrange

from tqdm import tqdm
import wandb
# from transfusion_pytorch import Transfusion, print_modality_sample


torch.cuda.empty_cache()

sys.path.insert(0,'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch')


from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    print_modality_sample,
    exists
)


def divisible_by(num, den):
    return (num % den) == 0

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
      print(f"Checkpoint file {ckpt_dir} does not exist.")
      return state
  
  checkpt = torch.load(ckpt_dir, map_location=device)
  state['model'].load_state_dict(checkpt['model'].state_dict(), strict=False)
  state['step'] = checkpt['step']
  state['epoch'] = checkpt['epoch']
  print(f'loaded checkpoint from {ckpt_dir}')
  return state

def save_checkpoint_for_non_ddp(save_path,state):
    model = state['model']
    print(f'isinstance(model, torch.nn.parallel.DistributedDataParallel):{isinstance(model, torch.nn.parallel.DistributedDataParallel) }')
    model_state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
    checkpoint = {
        'model': model_state,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved for non-DDP use at {save_path}")

def save_checkpoint(ckpt_dir, state):
  torch.save(state,ckpt_dir)

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)

class JointDataset(Dataset):
    def __init__(self):
        self.directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_json_npy'
        self.filenames = [f for f in os.listdir(self.directory) if f.endswith('.pt')]
        self.transform = transforms.Compose([
            transforms.CenterCrop((28, 28)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        tok, im=torch.load(file_path)
        token = torch.tensor(tok, dtype=torch.long)[0]
        im = torch.tensor(im, dtype=torch.float)[0] 
        im_t = self.transform(im).unsqueeze(0)
      
        return token,im_t



def create_joint_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = JointDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def init_distributed(rank,local_rank,ws,address,port):
  dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)

  torch.cuda.set_device(local_rank)
  print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong



def train_modality():
    print("train moidalitiy.............")

    world_size=int(os.environ["SLURM_NTASKS"])
    local_rank=int(os.environ["SLURM_LOCALID"])
    rank=int(os.environ["SLURM_PROCID"])
    address=os.environ["MASTER_ADDR"]
    port=os.environ["MASTER_PORT"]
    print(f"world size and rank:{world_size}, {rank}")

    #init_distributed(rank,local_rank,world_size, address, port)
    dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device= torch.device('cuda', local_rank)
    print(f"Process {rank} using device: {device}")

  #...........................encoder -decoder..............................
    dataset =  JointDataset()
    sampler = DistributedSampler(dataset, shuffle = True)
    autoencoder_dataloader = DataLoader(dataset, batch_size=12, sampler = sampler, shuffle=False)
    autoencoder_iter_dl = cycle(autoencoder_dataloader)
    autoencoder_train_steps = 5000
    dim_latent = 32
    num_epochs=100

    enc = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding = 1),
        nn.Conv2d(4, 8, 4, 2, 1),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Conv2d(8, dim_latent, 1),
        Rearrange('b d ... -> b ... d'),
        Normalize()
    ).to(device)
    encoder = DDP(enc, device_ids = [local_rank])

    dec = nn.Sequential(
        Rearrange('b ... d -> b d ...'),
        nn.Conv2d(dim_latent, 8, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 4, 4, 2, 1),
        nn.Conv2d(4, 1, 3, padding = 1),
    ).to(device)
    decoder = DDP(dec, device_ids = [local_rank])

    # train autoencoder
    print('training autoencoder')
    # encoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder_checkpoint.pth')
    # decoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder_checkpoint.pth')
    # encoder.load_state_dict(encoder_checkpoint)
    # decoder.load_state_dict(decoder_checkpoint)
    autoencoder_optimizer = Adam([*encoder.parameters(), *decoder.parameters()], lr = 10e-6)
    if rank==0:
        wandb.init( project="transfusion")

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for step in range(autoencoder_train_steps):
            _, images = next(autoencoder_iter_dl)
            images = images.to(device)
            latents = encoder(images)
            latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.2) # add a bit of noise to latents
            reconstructed = decoder(latents)
            reconstructed=reconstructed.to(device)
            loss = F.mse_loss(images, reconstructed)
            loss.backward()
            autoencoder_optimizer.step()
            autoencoder_optimizer.zero_grad()
            if rank==0: 
                if step%100==0:
                    print("epcho step glo_step loss lr.............................: ",epoch, step, loss.item())
                wandb.log({"step": step, "train_loss": loss.item()})

        torch.save(encoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder28_checkpoint.pth')
        torch.save(decoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder28_checkpoint.pth')

if __name__=="__main__":
    train_modality()