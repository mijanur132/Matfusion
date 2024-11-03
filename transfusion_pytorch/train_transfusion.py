from functools import partial
from copy import deepcopy
import sys
import json
import torch
import os
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

    print("slrum ntask:", os.getenv("SLURM_NTASKS"))

    x=0  #so that code does not go into slurm_ntasks loop while running without slurm. Remove this before submitting to slurm. 
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
    print("device.....", device)
    print(f"Process {rank} using device: {device}")

    if device.type == 'cuda':
        print("GPU is available")
    else:
        print("GPU is not available, using CPU")

    directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/train'
    #dataloader = create_image_dataloader(directory, batch_size=1)
    dataloader, sampler = create_image_dataloader_ddp(directory, batch_size = 12)

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (384, 3), #384 for text, 3 for image
        #dim_latent = (384,192),
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = False
        )
    )
   
    model = model.to(device)
    #model = DDP(model, device_ids=[local_rank])
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    warmup_steps=1

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return 0.001
        if current_step > 3:
            return 0.001
        return 1.0

    optimizer = optim.Adam(model.parameters(), lr=10e-8)  # target lr
    scheduler = LambdaLR(optimizer, lr_lambda)

    num_epochs=100
    global_max = 0

    scaler = GradScaler()

    if rank==0:
        wandb.init( project="transfusion")
    
    glob_step = 0
    accum_itr = 10
    for epoch in range(num_epochs):
        sampler.set_epoch (epoch)
        for step, _images in enumerate(dataloader):
            glob_step+=1
            #optimizer.zero_grad()
            #images = randn(2, 192, 8, 8)
            images = _images.to(device)
            im_mean = images.mean()
            loss = model(images, return_loss = True, modality_type = 1)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            # Clip gradients: parameters, max_norm, norm_type
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)  #grad clipping
            if ( (step+1)% accum_itr == 0 or (step+1) == len(dataloader) ):
                optimizer.step()
                optimizer.zero_grad()
                #scheduler.step(glob_step)
            if rank==0 and step%100==0:
                print("epcho step loss lr.............................: ",epoch, glob_step, loss.item(),  optimizer.param_groups[0]['lr'])
            if rank==0:
                wandb.log({"step": step, "train_loss": loss.item(), "mean": im_mean.item()})
            if torch.isnan(loss):
                break
        scheduler.step(epoch)

    dist.destroy_process_group()

def train_transfusion():
    x=0  #so that code does not go into slurm_ntasks loop while running without slurm. Remove this before submitting to slurm. 
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
    if device.type == 'cuda':
        print("GPU is available")
    else:
        print("GPU is not available, using CPU")

    use_flex_attn = True

    text_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json'
    text_dataloader, text_sampler = create_dataloader_ddp(text_directory, batch_size=1)
  
    directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/train'
    dataloader, sampler = create_image_dataloader_ddp(directory, batch_size = 1)

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
    if use_flex_attn: model = model.cuda()
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=10e-8)  # target lr
    num_epochs=100
    global_max = 0
    scaler = GradScaler()
    glob_step = 0
    accum_itr = 1
    if rank==0:
        wandb.init( project="transfusion")
    print("dataloader length:", len(joint_dataloader))
    for epoch in range(num_epochs):
        sampler.set_epoch (epoch)
        for step, (_tokens,_images) in enumerate(joint_dataloader):
            # print("tokens:", _tokens[0][0])
            # print("images:", _images)
            optimizer.zero_grad()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)  #grad clipping
            #if ((step+1)% accum_itr == 0 or (step+1) == len(dataloader)):
            optimizer.step()
      
            if rank == 0 and step%1 == 0:
                print("epoch step loss lr.............................: ",epoch, glob_step, loss.item(),  optimizer.param_groups[0]['lr'])
            if rank==0:
                wandb.log({"step": step, "train_loss": loss.item()})
            if torch.isnan(loss):
                break

    dist.destroy_process_group()


def train_transfusion_dummy():
    x=0  #so that code does not go into slurm_ntasks loop while running without slurm. Remove this before submitting to slurm. 
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
    if device.type == 'cuda':
        print("GPU is available")
    else:
        print("GPU is not available, using CPU")

    use_flex_attn = True

    text_directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_json'
    text_dataloader, text_sampler = create_dataloader_ddp(text_directory, batch_size=1)
    #text_dataloader = create_dataloader(text_directory, batch_size=2)

    directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/numpy_files/train'
    dataloader, sampler = create_image_dataloader_ddp(directory, batch_size = 1)
    model = Transfusion(
        num_text_tokens = 6,
        dim_latent = (8), # specify multiple latent dimensions, one for each modality
        #channel_first_latent = True,
        #modality_default_shape = ((32, 32)),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )
    if use_flex_attn: model = model.cuda()
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=10e-6)  # target lr
    num_epochs=100
    global_max = 0
    scaler = GradScaler()
    glob_step = 0
    accum_itr = 1
    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)


    if rank==0:
        wandb.init( project="transfusion")
    for epoch in range(num_epochs):
        sampler.set_epoch (epoch)
        for step, (_tokens,_images) in enumerate(zip(text_dataloader, dataloader)):
            optimizer.zero_grad()
            glob_step+=1
            zeros_tensor = torch.zeros(1, 1)  #batchsize
            normalized_tokens = torch.cat((zeros_tensor,_tokens), dim = 1)
            normalized_tokens = normalized_tokens.cuda().long()
            nt = normalized_tokens.squeeze()
            images = _images.to(device)
            im = images.squeeze()
            inp = [[nt, (1, im)]]
        #     inp = [[randint_((16,)), (0, randn(4, 384)), randint_((8,)), (1, randn(6, 192))],
        # [randint_((16,)), randn(7, 384), randint_((5,)), (1, randn(2, 192)), randint_((9,))]]
            inp = [[randint_((8,)), (0, randn(2, 8))]]
            loss = model(inp, return_loss = True)#, modality_type = 1)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)  #grad clipping
            #if ((step+1)% accum_itr == 0 or (step+1) == len(dataloader)):
            optimizer.step()
      
            if rank == 0 and step%1 == 0:
                print("epoch step loss lr.............................: ",epoch, glob_step, loss.item(),  optimizer.param_groups[0]['lr'])
            if rank==0:
                wandb.log({"step": step, "train_loss": loss.item()})
            if torch.isnan(loss):
                break

    dist.destroy_process_group()




if __name__=="__main__":
    #train()
    #train_modality()
    train_transfusion()
    #train_transfusion_dummy()