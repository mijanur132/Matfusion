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


class TokenDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.json')]
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        with open(file_path, 'r') as file:
            data = json.load(file)
        tokens = data['input_ids'][0]
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32)
        return tokens_tensor

class ImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.npy')]
        self.transform = transforms.CenterCrop((64, 64))

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        data = np.load(file_path)
        data_tensor = torch.from_numpy(data).float()
        data_tensor = self.transform(data_tensor)
        return data_tensor


#text dataloader
def create_dataloader(directory, batch_size=1, shuffle=True):  
    dataset = TokenDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def create_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = TokenDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def create_image_dataloader(directory, batch_size=1, shuffle=True):
    dataset = ImageDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader

def create_image_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = ImageDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def create_joint_dataloader_ddp(directory, batch_size=1, shuffle=True):
    dataset = JointDataset(directory)
    sampler = DistributedSampler(dataset, shuffle = shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader, sampler

def train_text():
    # Usage
    directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_text'
    dataloader = create_dataloader(directory, batch_size=2, transform=None)

    use_flex_attn = True
    return_loss = True

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        print("skipping...")

    model = Transfusion(
        num_text_tokens = 30000,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Learning rate might need tuning

    if use_flex_attn:
        model = model.cuda()

    # To iterate over data in your training loop
    num_epochs=100
    global_max = 0
    for epoch in range(num_epochs):
        for step, _tokens in enumerate(dataloader):
            optimizer.zero_grad()
            global_max= max(torch.max(_tokens),global_max)
            print("global max:.....",global_max)
            zeros_tensor = torch.zeros(2, 1)
            normalized_tokens = torch.cat((zeros_tensor,_tokens), dim = 1)
            #print(normalized_tokens.size(),normalized_tokens[0])
            normalized_tokens = normalized_tokens.cuda().long()
            loss = model(normalized_tokens, return_loss = return_loss)
            loss.backward()
            optimizer.step()
            print("epcho step loss: ",epoch, step, loss.item())
           # wandb.log({"step": step, "train_loss": loss.item()})
            if torch.isnan(loss):
                break

def init_distributed(rank,local_rank,ws,address,port):
  dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=ws)

  torch.cuda.set_device(local_rank)
  print("***************rank and world size*****************:",dist.get_rank(), dist.get_world_size()) ### most like wrong


def train_modality():
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

    #init_distributed(rank,local_rank,world_size, address, port)
    dist.init_process_group(backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device= torch.device('cuda', local_rank)
    print(f"Process {rank} using device: {device}")

    directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/combined'
    dataloader, sampler = create_image_dataloader_ddp(directory, batch_size = 12)
    model = Transfusion(
        num_text_tokens = 8,
        dim_latent = (64,), 
        #dim_latent = (384,192),
        channel_first_latent = True,
        modality_default_shape = (64,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = False
        )
    )

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    state = dict( model = model, step=0, epoch=0)
    checkpoint_dir = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mod_only'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    initial_step = int(state['step'])
    initial_epoch = int(state['epoch'])    

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
        if rank==0:
            print("len dataloader:", len(dataloader))
            print(f"latest checkpoint.................:{latest_checkpoint}")
            checkpoint_dir_temp = os.path.join(checkpoint_dir, latest_checkpoint)
            state = restore_checkpoint(checkpoint_dir_temp, state, device)
            initial_epoch = int(state['epoch'])+1
            initial_step = int(state['step'])+1
            print("initial_epoch:", initial_epoch)
        else:
            latest_checkpoint = None
            print("No checkpoint files found..........")

    warmup_steps=1

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return 0.001
        if current_step > 3:
            return 0.001
        return 1.0

    optimizer = optim.Adam(model.parameters(), lr=10e-8)  # target lr
    #scheduler = LambdaLR(optimizer, lr_lambda)

    num_epochs=100
    global_max = 0

    scaler = GradScaler()

    if rank==0:
        wandb.init( project="transfusion")
    
    glob_step = 0
    accum_itr = 1
    for epoch in range(num_epochs):
        sampler.set_epoch (epoch)
        optimizer.zero_grad()
        for step, _images in enumerate(dataloader):
            glob_step+=1
            #optimizer.zero_grad()
            #images = randn(2, 192, 8, 8)
            images = _images.to(device)
            im_mean = images.mean()
            loss = model(images, return_loss = True) #, modality_type = 1)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            # Clip gradients: parameters, max_norm, norm_type
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)  #grad clipping
            if ( (step+1)% accum_itr == 0 or (step+1) == len(dataloader) ):
                optimizer.step()
                optimizer.zero_grad()
            if rank==0: 
                if step%100==0:
                    print("epcho step glo_step loss lr.............................: ",epoch, step, glob_step, loss.item(),  optimizer.param_groups[0]['lr'])
                wandb.log({"step": step, "train_loss": loss.item(), "mean": im_mean.item()})

        if (epoch>0 and epoch%1==0) and rank==0:
            state['epoch']=epoch
            state['step']=step
            #save_checkpoint_for_non_ddp(os.path.join(checkpoint_dir, f'non_ddp_checkpoint_{epoch}_{step}.pth'),state)
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_mod_{epoch}_{step}.pth'), state)
            print(f'chepoint saved: checkpoint_{epoch}_{step}.pth')
            # one_multimodal_sample = model.module.sample()
            # save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/mod_sample_out_{epoch}.pt"
            # torch.save(one_multimodal_sample,save_path)
       # scheduler.step(epoch)

    dist.destroy_process_group()

def train_transfusion():
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

    joint_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/comb_json_npy'
    joint_dataloader, joint_sampler = create_joint_dataloader_ddp(joint_directory, batch_size=1)
    model = Transfusion(
        num_text_tokens = 30000,
        dim_latent = (128), # specify multiple latent dimensions, one for each modality
        channel_first_latent = False,
        #modality_default_shape = ((32,), (32,)),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = False
        )
    )
    # if use_flex_attn: model = model.cuda()
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

    num_epochs=100
    scaler = GradScaler()
    glob_step = 0
    accum_itr = 4
    if rank==0:
        wandb.init( project="transfusion")
    print("dataloader length:", len(joint_dataloader))
    for epoch in range(initial_epoch, num_epochs):
        print("epoch:",epoch)
        joint_sampler.set_epoch (epoch)
        optimizer.zero_grad()
        for step, (_tokens,_images) in enumerate(joint_dataloader):
            glob_step+=1
            zeros_tensor = torch.zeros(1, 1)  #batchsize
            normalized_tokens = torch.cat((zeros_tensor,_tokens), dim = 1)
            normalized_tokens = normalized_tokens.cuda().long()
            nt = normalized_tokens.squeeze()
            images = _images.to(device)
            im = images.squeeze()
            inp = [[nt, (0, im)]]
            loss = model(inp, return_loss = True)#, modality_type = 1)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)  #grad clipping
            if ((step+1)% accum_itr == 0 or (step+1) == len(joint_dataloader) ):
                optimizer.step()
                optimizer.zero_grad()
            if rank == 0 and step%1 == 0:
                print("epoch step loss lr.............................: ",epoch, glob_step, loss.item(),  optimizer.param_groups[0]['lr'])
                wandb.log({"glob_step": glob_step, "train_loss": loss.item()})
        if (epoch>0 and epoch%1==0) and rank==0:
            state['epoch']=epoch
            state['step']=step
            save_checkpoint_for_non_ddp(os.path.join(checkpoint_dir, f'non_ddp_checkpoint_{epoch}_{step}.pth'),state)
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch}_{step}.pth'), state)
            print(f'chepoint saved: checkpoint_{epoch}_{step}.pth')
      
            one_multimodal_sample = model.module.sample()
            save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/sample_out_{epoch}.pt"
            torch.save(one_multimodal_sample,save_path)
            # from transformers import BertTokenizer
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # #decoded_text = tokenizer.decode(token_ids)
            
    dist.destroy_process_group()



if __name__=="__main__":
    #train()
    train_modality()
    #train_transfusion()
    #train_transfusion_dummy()