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
            transforms.CenterCrop((64, 64)),
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
        self.transform = transforms.Compose([
            transforms.CenterCrop((64, 64)),
            #transforms.Lambda(lambda x: x * 25500)
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])

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
    print("train moidalitiy.............")
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

  #...........................encoder -decoder..............................
    dataset =  JointDataset()
    autoencoder_train_steps = 5000
    dim_latent = 32
    num_epochs=100
    global_max = 0
    glob_step = 0
    accum_itr = 1

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
    print('training autoencoder')
    # encoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder_checkpoint.pth')
    # decoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder_checkpoint.pth')
    # encoder.load_state_dict(encoder_checkpoint)
    # decoder.load_state_dict(decoder_checkpoint)
    autoencoder_optimizer = Adam([*encoder.parameters(), *decoder.parameters()], lr = 3e-4)
    
    autoencoder_dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
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

    torch.save(encoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder_checkpoint.pth')
    torch.save(decoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder_checkpoint.pth')

    # transfusion

    model = Transfusion(
        num_text_tokens = 20,
        dim_latent = dim_latent,
        modality_default_shape = (128, 128),
        modality_encoder = encoder,
        modality_decoder = decoder,
        add_pos_emb = True,
        modality_num_dim = 2,
        transformer = dict(
            dim = 64,
            depth = 4,
            dim_head = 32,
            heads = 8
        )
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # training transfusion
    def collate_fn(data):
        data = [*map(list, data)]
        return data

    sampler = DistributedSampler(dataset, shuffle = True)
    dataloader = DataLoader(dataset, batch_size=12, sampler = sampler, collate_fn = collate_fn, shuffle=False)
    iter_dl = cycle(dataloader)
    optimizer = Adam(model.module.parameters_without_encoder_decoder(), lr = 3e-4)

    state = dict( model = model, step=0, epoch=0)
    checkpoint_dir = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mod_28'
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
        print("No checkpoint files found..................")

    if rank==0:
        wandb.init( project="transfusion")

    for epoch in range(initial_epoch,num_epochs):
        sampler.set_epoch (epoch)
        optimizer.zero_grad()
        for step in range(len(dataloader)):
            glob_step+=1
            #optimizer.zero_grad()
            images = next(iter_dl)
            loss = model(images)
            loss = loss/accum_itr   #grad accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if ( (step+1)% accum_itr == 0 or (step+1) == len(dataloader) ):
                optimizer.step()
                optimizer.zero_grad()
            if rank==0: 
                if step%100==0:
                    print("epcho step glo_step loss lr.............................: ",epoch, step, glob_step, loss.item())
                wandb.log({"step": step, "train_loss": loss.item()})

        if (epoch>0 and epoch%1==0) and rank==0:
            state['epoch']=epoch
            state['step']=step
            #save_checkpoint_for_non_ddp(os.path.join(checkpoint_dir, f'non_ddp_checkpoint_{epoch}_{step}.pth'),state)
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_mod_{epoch}_{step}.pth'), state)
            print(f'chepoint saved: checkpoint_{epoch}_{step}.pth')

            save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample"
            one_multimodal_sample = model.module.sample(max_length = 10)
            print_modality_sample(one_multimodal_sample)
            if len(one_multimodal_sample) < 2:
                continue
            maybe_label, maybe_image, *_ = one_multimodal_sample
            print("maybelabel:", maybe_label)
            filename = f'{save_path}/{epoch}_{step}.png'
            print(filename)
            save_image(
                maybe_image[1].cpu().clamp(min = 0., max = 1.),
                filename,
            )

    dist.destroy_process_group()



def train_mnist():
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

    directory = '/lustre/orion/stf218/proj-shared/brave/brave_database/mnist'
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    #if rank == 0:
    train_data = MNIST(root = directory, train=True, download=True, transform=transform)
    dataloader = DataLoader(train_data, batch_size=64, shuffle=True)   

    model = Transfusion(
        num_text_tokens = 8,
        dim_latent = (28,), 
        #dim_latent = (384,192),
        channel_first_latent = True,
        modality_default_shape = (28,),
        transformer = dict(
            dim = 512,
            depth = 4,
            use_flex_attn = False
        )
    )

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    state = dict( model = model, step=0, epoch=0)
    checkpoint_dir = '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist_depth6'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    initial_step = int(state['step'])
    initial_epoch = int(state['epoch'])   


    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
        if rank==0:
            #print("len dataloader:", len(dataloader))
            print(f"latest checkpoint.................:{latest_checkpoint}")
            checkpoint_dir_temp = os.path.join(checkpoint_dir, latest_checkpoint)
            state = restore_checkpoint(checkpoint_dir_temp, state, device)
            initial_epoch = int(state['epoch'])+1
            initial_step = int(state['step'])+1
            print("initial_epoch:", initial_epoch)
        else:
            latest_checkpoint = None
            print("No checkpoint files found..........")

  
    optimizer = optim.Adam(model.parameters(), lr=10e-6)  # target lr
    num_epochs=100
    global_max = 0
    if rank==0:
        wandb.init( project="transfusion")
    
    glob_step = 0
    accum_itr = 5

    for epoch in range(initial_epoch,num_epochs):
        #sampler.set_epoch (epoch)
        optimizer.zero_grad()
        for step, (_images,labels) in enumerate(dataloader):
            glob_step+=1
            #optimizer.zero_grad()
            #images = randn(2, 192, 8, 8)
            images = _images.to(device).float()
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
                    print("epcho step glo_step loss lr.............................: ",epoch, step, glob_step, loss.item(),  optimizer.param_groups[0]['lr'], images.min(),images.max())
                wandb.log({"step": step, "train_loss": loss.item(), "mean": im_mean.item()})

        if (epoch>0 and epoch%1==0) and rank==0:
            state['epoch']=epoch
            state['step']=step
            #save_checkpoint_for_non_ddp(os.path.join(checkpoint_dir, f'non_ddp_checkpoint_{epoch}_{step}.pth'),state)
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_mnist1_128_{epoch}_{step}.pth'), state)
            print(f'chepoint saved: checkpoint_mnist_{epoch}_{step}.pth')
            prime = [tensor(model.module.som_ids[0])]
            one_multimodal_sample = model.module.sample(prime, max_length = 4, cache_kv = True)
            save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/mod_mnist_14nov_{epoch}.pt"
            torch.save(one_multimodal_sample,save_path)
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
        if (epoch>0 and epoch%3==0) and rank==0:
            state['epoch']=epoch
            state['step']=step
            save_checkpoint_for_non_ddp(os.path.join(checkpoint_dir, f'non_ddp_checkpoint_{epoch}_{step}.pth'),state)
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch}_{step}.pth'), state)
            print(f'chepoint saved: checkpoint_{epoch}_{step}.pth')
      
            # one_multimodal_sample = model.module.sample()
            # save_path = f"/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/transfusion_pytorch/output_sample/sample_out_{epoch}.pt"
            # torch.save(one_multimodal_sample,save_path)
            # from transformers import BertTokenizer
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # #decoded_text = tokenizer.decode(token_ids)
            
    dist.destroy_process_group()



if __name__=="__main__":
    #train()
    #train_transfusion()
    #train_transfusion_dummy()
    #train_mnist()
    train_modality()