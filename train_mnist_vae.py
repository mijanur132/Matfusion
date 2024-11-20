from shutil import rmtree
from pathlib import Path

import torch
from torch import nn, tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange
from einops.layers.torch import Rearrange

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as T
from torchvision.utils import save_image

from tqdm import tqdm
from transfusion_pytorch import Transfusion, print_modality_sample

import os


results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# functions

def divisible_by(num, den):
    return (num % den) == 0

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

# dataset

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data/mnist',
            download = True
        )   

        self.transform = T.Compose([
            T.PILToTensor(),
            T.RandomResizedCrop((28, 28), scale = (0.8, 1.))
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = self.transform(pil)   #1,28,28
        return tensor(labels), (digit_tensor / 255).float()

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
        token = torch.tensor(tok, dtype=torch.long)[0:5]
        im = torch.tensor(im, dtype=torch.float)[0] 
        im_t = self.transform(im).unsqueeze(0)
      
        return token,im_t


dataset =  JointDataset()
#dataset =  MnistDataset()

# contrived encoder / decoder with layernorm at bottleneck

autoencoder_train_steps = 15
dim_latent = 16

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)

encoder = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding = 1),
    nn.Conv2d(4, 8, 4, 2, 1),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Conv2d(8, dim_latent, 1),
    Rearrange('b d ... -> b ... d'),
    Normalize()
).cuda()

decoder = nn.Sequential(
    Rearrange('b ... d -> b d ...'),
    nn.Conv2d(dim_latent, 8, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4, 2, 1),
    nn.Conv2d(4, 1, 3, padding = 1),
).cuda()

# train autoencoder

encoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder_checkpoint.pth')
decoder_checkpoint = torch.load('/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder_checkpoint.pth')

encoder.load_state_dict(encoder_checkpoint)
decoder.load_state_dict(decoder_checkpoint)


autoencoder_optimizer = Adam([*encoder.parameters(), *decoder.parameters()], lr = 3e-4)
autoencoder_dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

autoencoder_iter_dl = cycle(autoencoder_dataloader)

print('training autoencoder')

with tqdm(total = autoencoder_train_steps) as pbar:
    for _ in range(autoencoder_train_steps):
        _, images = next(autoencoder_iter_dl)
        images = images.cuda()
        #print(images.shape)

        latents = encoder(images)
        latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.2) # add a bit of noise to latents
        reconstructed = decoder(latents)

        loss = F.mse_loss(images, reconstructed)

        loss.backward()

        pbar.set_description(f'loss: {loss.item():.5f}')

        autoencoder_optimizer.step()
        autoencoder_optimizer.zero_grad()

        pbar.update()

# torch.save(encoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/encoder_checkpoint.pth')
# torch.save(decoder.state_dict(), '/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch/checkpoints/mnist/decoder_checkpoint.pth')

# transfusion

model = Transfusion(
    num_text_tokens = 10,
    dim_latent = dim_latent,
    modality_default_shape = (14, 14),
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
).cuda()

# training transfusion

def collate_fn(data):
    data = [*map(list, data)]
    return data

dataloader = DataLoader(dataset, batch_size = 16, collate_fn = collate_fn, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters_without_encoder_decoder(), lr = 3e-4)

# train loop

transfusion_train_steps = 25_000

print('training transfusion with autoencoder')

with tqdm(total = transfusion_train_steps) as pbar:
    for index in range(transfusion_train_steps):
        step = index + 1

        model.train()
        x= next(iter_dl)
        #print(x[0][0],x[0][1]) #torch.tensor(1), torch.Size([1, 28, 28])
        loss = model(x)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f'loss: {loss.item():.3f}')

        pbar.update()

        # eval

        if divisible_by(step, 500):
            one_multimodal_sample = model.sample(max_length = 10)

            print_modality_sample(one_multimodal_sample)

            if len(one_multimodal_sample) < 2:
                continue

            maybe_label, maybe_image, *_ = one_multimodal_sample
            print("maybelabel:", maybe_label)
            filename = f'{step}.{maybe_label[1].item()}.png'

            save_image(
                maybe_image[1].cpu().clamp(min = 0., max = 1.),
                str(results_folder / filename),
            )