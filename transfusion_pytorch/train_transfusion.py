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
from torch.amp import autocast, GradScaler
from torchinfo import summary
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR

torch.cuda.empty_cache()
sys.path.insert(0,'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch')

from torch import nn, randint, randn, tensor, cuda
from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    exists
)

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


class JointDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.pt')]
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        tok, im=torch.load(file_path)
        token = torch.tensor(tok, dtype=torch.long)
        image = torch.from_numpy(im).float()
        return token,image


def create_joint_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = JointDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def init_distributed(rank,local_rank,ws,address,port):
  dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)

  torch.cuda.set_device(local_rank)
  print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong


def sample_transfusion():
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

    #joint_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_json_npy'
    #joint_dataloader, joint_sampler = create_joint_dataloader_ddp(joint_directory, batch_size=1)
    model = Transfusion(
        num_text_tokens = 30000,
        dim_latent = (128), # specify multiple latent dimensions, one for each modality
        channel_first_latent = False,
        modality_default_shape = (64,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = False
        )
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=10e-9)  # target lr
    state = dict( model = model, step=0, epoch=0)

    checkpoint_dir = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    initial_step = int(state['step'])
    initial_epoch = int(state['epoch'])    

    if checkpoint_files:
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
    if rank == 0:
        prime = [tensor(model.module.som_ids[0])]
        #one_multimodal_sample = model.module.sample(prime, max_length = 2048, cache_kv = True)
        one_multimodal_sample = model.module.sample( max_length = 2048, cache_kv = True)
        save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/sample_out_test_1.pt"
        torch.save(one_multimodal_sample,save_path)
        
  

if __name__=="__main__":
    sample_transfusion()
