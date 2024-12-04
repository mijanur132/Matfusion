from functools import partial
from copy import deepcopy
import sys
import json
import torch
import os
import glob
import wandb
import torch.optim as optim
import numpy as np

import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import MNIST
from transformers import BertTokenizer
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image
import argparse


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.cuda.empty_cache()
sys.path.insert(0,'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch')

from torch import nn, randint, randn, tensor, cuda

from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    print_modality_sample,
    exists
)

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)
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

def save_checkpoint(ckpt_dir, state):
  torch.save(state,ckpt_dir)

class JointDataset(Dataset):
    def __init__(self, directory, dim_latent):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.pt')]
        self.transform = transforms.Compose([
            transforms.CenterCrop((dim_latent, dim_latent)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        label, im=torch.load(file_path)
        label = torch.tensor(label, dtype=torch.long)#[0:50]
        image = torch.from_numpy(im).float()[0]
        image = self.transform(image)
        return ([label,image])

class AEdataset(Dataset):
    def __init__(self):
        self.directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_json_npy'
        self.filenames = [f for f in os.listdir(self.directory) if f.endswith('.pt')]
        self.transform = transforms.Compose([
            transforms.CenterCrop((64, 64)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        tok, im=torch.load(file_path)
        token = torch.tensor(tok, dtype=torch.long)
        zero_tensor = torch.tensor([0], dtype=torch.long)  # Create a tensor with a single zero
        token = torch.cat((token, zero_tensor), dim=0)  # Concatenate the zero tensor to the token tensor
        token = token.cuda().long()
        # nt = normalized_tokens.squeeze()

        im = torch.tensor(im, dtype=torch.float)[0]
        im_t = self.transform(im)#.unsqueeze(0)
        return token,im_t

def create_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = TokenDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def create_image_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = ImageDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def create_joint_dataloader_ddp(directory, dim_latent = 28, batch_size=1, shuffle=True):
    dataset = JointDataset(directory, dim_latent)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def init_distributed(rank,local_rank,ws,address,port):
  dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)

  torch.cuda.set_device(local_rank)
  print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong


def train_transfusion(_dim_latent, _mod_shape, _xdim, _xdepth):
    dim_latent, mod_shape, xdim, xdepth = _dim_latent, _mod_shape, _xdim, _xdepth
    x=1  #so that code does not go into slurm_ntasks loop while running without slurm. Remove this before submitting to slurm. 
    if x and "SLURM_NTASKS" in os.environ:
        print("should not come here")
        world_size=int(os.environ["SLURM_NTASKS"])
        local_rank=int(os.environ["SLURM_LOCALID"])
        rank=int(os.environ["SLURM_PROCID"])
        address=os.environ["MASTER_ADDR"]
        port=os.environ["MASTER_PORT"]
        print(f"world size and rank:{world_size}, {rank}")
    else:
        world_size= torch.cuda.device_count()
        rank=int(os.getenv('RANK', 0))
        local_rank= int(os.getenv('LOCAL_RANK', 0))
        address="127.0.0.1"
        port=29500
        print(f"world size and rank:{world_size}, {rank}, {local_rank}")
        
    dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device= torch.device('cuda', local_rank)
    print(f"Process {rank} using device: {device}")

    joint_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_sg_npy'
    joint_dataloader, joint_sampler = create_joint_dataloader_ddp(joint_directory, dim_latent, batch_size=16)
    iter_dl = cycle(joint_dataloader)
    dataset =  AEdataset()
    autoencoder_train_steps = 500

    encoder = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding = 1),
        nn.Conv2d(4, 8, 4, 2, 1),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Conv2d(8, dim_latent, 1),
        Rearrange('b d ... -> b ... d'),
        Normalize()
    ).to(device)

    decoder = nn.Sequential(
        Rearrange('b ... d -> b d ...'),
        nn.Conv2d(dim_latent, 8, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 4, 4, 2, 1),
        nn.Conv2d(4, 1, 3, padding = 1),
    ).to(device)


    # train autoencoder
    loadckpt = 1
    if loadckpt:
        encoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder1_checkpoint.pth')
        decoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder1_checkpoint.pth')
        enc_new_state_dict = {k.replace('module.', ''): v for k, v in encoder_checkpoint.items()}
        dec_new_state_dict = {k.replace('module.', ''): v for k, v in decoder_checkpoint.items()}
        encoder.load_state_dict(enc_new_state_dict)
        decoder.load_state_dict(dec_new_state_dict)
    else: 
        print('training autoencoder')
        autoencoder_optimizer = Adam([*encoder.parameters(), *decoder.parameters()], lr = 3e-4)
        autoencoder_dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)
        autoencoder_iter_dl = cycle(autoencoder_dataloader)
        with tqdm(total = autoencoder_train_steps) as pbar:
            for _ in range(autoencoder_train_steps):
                _, images = next(autoencoder_iter_dl)
                images = images.cuda()
                latents = encoder(images)
                latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.2) # add a bit of noise to latents
                reconstructed = decoder(latents)
                loss = F.mse_loss(images, reconstructed)
                loss.backward()
                pbar.set_description(f'loss: {loss.item():.5f}')
                autoencoder_optimizer.step()
                autoencoder_optimizer.zero_grad()
                pbar.update()

        torch.save(encoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder1_checkpoint.pth')
        torch.save(decoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder1_checkpoint.pth')

    model = Transfusion(
        num_text_tokens = 50000,
        dim_latent = dim_latent,
        modality_default_shape = (mod_shape,mod_shape),
        modality_encoder = encoder,
        modality_decoder = decoder,
        add_pos_emb = True,
        modality_num_dim = 2,
        transformer = dict(
            dim = xdim,
            depth = xdepth,
            dim_head = 32,
            heads = 8
        )
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    ema_model = model.module.create_ema().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=3e-4)  
    optimizer = Adam(model.module.parameters_without_encoder_decoder(), lr = 3e-4)

    state = dict( model = model, step=0, epoch=0)
    checkpoint_dir = f'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/matsci_{dim_latent}_{mod_shape}_{xdim}_{xdepth}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    initial_step = int(state['step'])
    initial_epoch = int(state['epoch'])    
    want_checkpt = 0
    if checkpoint_files and want_checkpt:
        latest_checkpoint = checkpoint_files[0]
        if rank==0:
            print(f"latest checkpoint.................:{latest_checkpoint}")
            checkpoint_dir_temp = os.path.join(checkpoint_dir, latest_checkpoint)
            state = restore_checkpoint(checkpoint_dir_temp, state, device)
            initial_epoch = int(state['epoch'])+1
            initial_step = int(state['step'])+1
            print("initial_epoch:", initial_epoch)
        else:
            latest_checkpoint = None
            print("No checkpoint files found..........")

    num_epochs=100
    scaler = GradScaler()
    glob_step = 0
    accum_itr = 1
    if rank==0:
        wandb.init( project="transfusion")

    print("dataloader length:", len(joint_dataloader))
    for epoch in range(initial_epoch, num_epochs):
        print("epoch:",epoch)
        joint_sampler.set_epoch (epoch)
        optimizer.zero_grad()
        for step in range(len(joint_dataloader)):
            glob_step+=1
            data = next(iter_dl)
            print(data)
            loss = model(data, return_loss = True)#, modality_type = 1)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if ((step+1)% accum_itr == 0 or (step+1) == len(joint_dataloader) ):
                optimizer.step()
                optimizer.zero_grad()
            if rank == 0:
                if step%1 == 0:
                    print("epoch step loss lr.............................: ",epoch, glob_step, loss.item())
                    wandb.log({"glob_step": glob_step, "train_loss": loss.item()})
                if step%500 == 0:
                    state['epoch']=epoch
                    state['step']=step
                    #save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_matsci_{epoch}_{step}.pth'), state)
                    print(f'chepoint saved: checkpoint_{epoch}_{step}.pth')
                    save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/matsci_{dim_latent}_{mod_shape}_{xdim}_{xdepth}/"
                    os.makedirs(save_path, exist_ok=True)
                    multimodal = 1
                    if multimodal: 
                        one_multimodal_sample = model.module.sample(max_length = 100)
                        print_modality_sample(one_multimodal_sample)
                        if len(one_multimodal_sample) >= 2:
                            maybe_label, maybe_image, *_ = one_multimodal_sample
                            filename = f'{save_path}/{epoch}_{step}_.png'
                            save_image(
                                maybe_image[1][1].cpu().clamp(min = 0., max = 1.),
                                filename
                                    )
    dist.destroy_process_group()



if __name__=="__main__":
    #train()
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument('--dim_latent', type=int, required=True, default = 16, help='Dimension of the latent space')
    parser.add_argument('--mod_shape', type=int, required=True, default = 14, help='Model shape as a list of integers')
    parser.add_argument('--xdim', type=int, required=True,  default = 256, help='Dimension x')
    parser.add_argument('--xdepth', type=int, required=True, default = 4, help='Depth x')
    

    args = parser.parse_args()
    dim_latent, mod_shape, xdim, xdepth = args.dim_latent, args.mod_shape, args.xdim, args.xdepth
    print(dim_latent, mod_shape, xdim, xdepth)


    train_transfusion(dim_latent, mod_shape, xdim, xdepth)
    #train_transfusion_dummy()
    #train_mnist()
    #train_modality()