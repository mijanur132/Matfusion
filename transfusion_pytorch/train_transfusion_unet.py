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
from shutil import rmtree

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.cuda.empty_cache()
sys.path.insert(0,'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch')

from torch import nn, randint, randn, tensor, cuda

from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    print_modality_sample,
    exists,
    TransfusionWithClassifier
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
  
  checkpt = torch.load(ckpt_dir, map_location=device, weights_only = True)
  state['model_state'] = checkpt ['model_state']
  state['step'] = checkpt['step']
  state['epoch'] = checkpt['epoch']
  print(f'loaded checkpoint from {ckpt_dir}')
  return state

def save_checkpoint(ckpt_dir, state):
  torch.save(state,ckpt_dir)

# def custom_collate_fn(data):
#     data = [*map(list, data)]
#     return data

class JointDataset(Dataset):
    def __init__(self, directory, dim_latent, split):
        self.directory = directory
        self.all_filenames = [f for f in os.listdir(directory) if f.endswith('.pt')]
        #np.random.shuffle(self.all_filenames) 
        self.transform = transforms.Compose([
            transforms.CenterCrop((dim_latent, dim_latent)),
            transforms.Lambda(lambda x: x**0.25)
           #transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])
        split_index = int(len(self.all_filenames) * 0.99)
        
        if split == 'train':
            self.filenames = self.all_filenames#[:split_index]
        else:
            self.filenames = self.all_filenames[split_index:]


    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        label, im=torch.load(file_path)
        label = torch.tensor(label, dtype=torch.long)#[0:50]
        image = torch.from_numpy(im)[0].float().unsqueeze(0)
        image = self.transform(image)
        return image, label


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

def create_joint_dataloader_ddp(directory, dim_latent = 28, batch_size=1, shuffle=True, split='train'):
    dataset = JointDataset(directory, dim_latent, split)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)#, collate_fn=custom_collate_fn)
    return dataloader, sampler

def init_distributed(rank,local_rank,ws,address,port):
  dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)

  torch.cuda.set_device(local_rank)
  print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong


def train_transfusion(_dim_latent, _mod_shape, _xdim, _xdepth):
    dim_latent, mod_shape, xdim, xdepth = _dim_latent, _mod_shape, _xdim, _xdepth
    CHANNEL_FIRST = True
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

    joint_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_sg_npy_top3'
    joint_dataloader, joint_sampler = create_joint_dataloader_ddp(joint_directory, dim_latent, batch_size=1, split= 'train')
    iter_dl = cycle(joint_dataloader)

    valid_dataloader, valid_sampler = create_joint_dataloader_ddp(joint_directory, dim_latent, batch_size=100, split= 'valid')
    valid_iter_dl = cycle(valid_dataloader)

    print("valid_len",len(valid_dataloader))

    class Encoder(Module):
        def forward(self, x):
            x = rearrange(x, '... 1 (h p1) (w p2) -> ... h w (p1 p2)', p1 = 2, p2 = 2)
            if CHANNEL_FIRST:
                x = rearrange(x, 'b ... d -> b d ...')

            return x * 2 - 1

    class Decoder(Module):
        def forward(self, x):

            if CHANNEL_FIRST:
                x = rearrange(x, 'b d ... -> b ... d')

            x = rearrange(x, '... h w (p1 p2) -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2)
            return ((x + 1) * 0.5).clamp(min = 0., max = 1.)
    
    old_model = Transfusion(
        num_text_tokens = 5,
        dim_latent = 4,
        modality_default_shape = (mod_shape,mod_shape),
        modality_encoder = Encoder(),
        modality_decoder = Decoder(),
        add_pos_emb = True,
        modality_num_dim = 2,
        channel_first_latent = CHANNEL_FIRST,
        transformer = dict(
            dim = xdim,
            depth = xdepth,
            dim_head = 32,
            heads = 8
        )
    )
    #old_model = model.to(device)
    #model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    #ema_model = model.module.create_ema().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=3e-6)  
    #optimizer = Adam(model.module.parameters_without_encoder_decoder(), lr = 3e-4)

    state = dict( model_state = old_model.state_dict(), step=0, epoch=0)
    checkpoint_dir = f'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/fnd_sg_unet_{dim_latent}_{mod_shape}_{xdim}_{xdepth}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    initial_step = int(state['step'])
    initial_epoch = int(state['epoch'])    
    want_checkpt = 1
    delete_chkpt =0
    if delete_chkpt:
        rmtree(checkpoint_dir, ignore_errors = True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    if checkpoint_files and want_checkpt:
        latest_checkpoint = checkpoint_files[0]
        if rank==0:
            print(f"latest checkpoint.................:{latest_checkpoint}")
            checkpoint_dir_temp = os.path.join(checkpoint_dir, latest_checkpoint)
            state = restore_checkpoint(checkpoint_dir_temp, state, device)
            old_model.load_state_dict(state['model_state'])
            print(old_model)
            initial_epoch = int(state['epoch'])+1
            initial_step = int(state['step'])+1
            print("initial_epoch:", initial_epoch)
        else:
            latest_checkpoint = None
            print("No checkpoint files found..........")

    for param in old_model.parameters():
        param.requires_grad = False

    model = TransfusionWithClassifier(
                                    old_model, 
                                    in_dim=dim_latent//2,   # or whatever the old model outputs
                                    out_dim=3   # number of classes
                                )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.module.classifier.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()


    num_epochs=100
    scaler = GradScaler()
    glob_step = 0
    accum_itr = 1
    if rank==0:
        wandb.init( project="transfusion")

    save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/fnd_sg_unet_{dim_latent}_{mod_shape}_{xdim}_{xdepth}/"
    rmtree(save_path, ignore_errors = True)
    os.makedirs(save_path, exist_ok=True)

    print("dataloader length:", len(joint_dataloader), len(valid_dataloader))
    for epoch in range(initial_epoch, num_epochs):
        print("epoch:",epoch)
        joint_sampler.set_epoch (epoch)
        optimizer.zero_grad()
        for step in range(len(joint_dataloader)):
            glob_step+=1
            images = next(iter_dl)
            labels = images[1]
            label_map = {1: 0, 3: 1, 14: 2}

            for old_label, new_label in label_map.items():
                labels[labels == old_label] = new_label
            labels= labels.to(device)
            output = model(images[0])#, return_loss = True)#, modality_type = 1)
            print(output)
            print(labels)
            loss = criterion(output, labels)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if ((step+1)% accum_itr == 0 or (step+1) == len(joint_dataloader) ):
                optimizer.step()
                optimizer.zero_grad()
               # ema_model.update()
            if rank == 0:
                if step%1 == 0:
                    print("epoch step loss lr.............................: ",epoch, glob_step, loss.item())
                    wandb.log({"glob_step": glob_step, "train_loss": loss.item()})
                if step >500 and step%500 == 0:
                    state['epoch']=epoch
                    state['step']=step

                    # rmtree(checkpoint_dir, ignore_errors = True)
                    # os.makedirs(checkpoint_dir, exist_ok=True)

                    #save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_classification_{epoch}_{step}.pth'), state)
                    print(f'chepoint saved: checkpoint_{epoch}_{step}.pth')
                  

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