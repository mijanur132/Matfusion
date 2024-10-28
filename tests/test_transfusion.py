import pytest
from functools import partial
from copy import deepcopy
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os

sys.path.insert(0,'/lustre/orion/stf218/proj-shared/brave/transfusion-pytorch')

from torch import nn, randint, randn, tensor, cuda

cuda_available = cuda.is_available()

from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    exists
)

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('use_flex_attn', (False, True))
def test_transfusion(
    cache_kv: bool,
    use_flex_attn: bool
):
    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = (384, 192), # specify multiple latent dimensions, one for each modality
        modality_default_shape = ((32,), (64,)),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    if use_flex_attn:
        model = model.cuda()

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    text_images_and_audio = [
        [randint_((16,)), (0, randn(4, 384)), randint_((5,)), (1, randn(6, 192))],
       # [randint_((16,)), randn(7, 384), randint_((5,)), (1, randn(2, 192)), randint_((5,))]
    ]

    # print("print([randint_((16,))]): ",[randint_((16,))])
    # print(" print([ (0, randn(4, 384))]):",[ (0, randn(4, 384))])
    # print("   print([ (1, randn(6, 192))]):",[ (1, randn(6, 192))])


    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 128, cache_kv = cache_kv)
    print("source:", text_images_and_audio[0])
    print("one_multimodal_sample:",one_multimodal_sample)


@pytest.mark.parametrize('use_flex_attn', (False, True))
def test_auto_modality_transform(
    use_flex_attn: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    text_and_images = [
        [randint_((16,)), randn(384, 2, 2), randint_((8,)), randn(384, 2, 2)],
        [randint_((16,)), randn(384, 2, 2), randint_((5,)), randn(384, 2, 2), randint_((9,))]
    ]

    loss = model(text_and_images)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 128)



class TokenDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Path to the directory with JSON files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.json')]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        with open(file_path, 'r') as file:
            data = json.load(file)
        tokens = data['input_ids'][0]
        
        # Assuming you want to normalize as discussed
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32)
        min_val = torch.min(tokens_tensor)
        max_val = torch.max(tokens_tensor)
        normalized_tokens = ((tokens_tensor - min_val) / (max_val - min_val)) * 256
        normalized_tokens = normalized_tokens.long()  # Convert to integer
        
        if self.transform:
            normalized_tokens = self.transform(normalized_tokens)
        
        return normalized_tokens


def create_dataloader(directory, batch_size=2, shuffle=True, transform=None):
    dataset = TokenDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def train():
    # Usage
    directory = '/path/to/dataset'
    dataloader = create_dataloader(directory, batch_size=32, transform=custom_transform)

    # To iterate over data in your training loop
    for normalized_tokens in dataloader:
        # Training logic here
        print(normalized_tokens.size())


@pytest.mark.parametrize('use_flex_attn', (False, True))
@pytest.mark.parametrize('return_loss', (False, True))

def test_text(
    use_flex_attn: bool,
    return_loss: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        print("skipping...")
        return pytest.skip()

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    if use_flex_attn:
        model = model.cuda()

    #text = randint(0, 256, (2, 1025))
    with open("/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_text/mp-23.cif.json", 'r') as file:
        data=json.load(file)
    tokens1=data['input_ids'][0]+[0]+data['input_ids'][0]
    with open("/lustre/orion/stf218/proj-shared/brave/brave_database/junqi_diffraction/token_text/mp-30.cif.json", 'r') as file:
        data=json.load(file)
    tokens2=data['input_ids'][0]+[0]+data['input_ids'][0]
    text=torch.stack([torch.tensor(tokens1),torch.tensor(tokens2)], dim=0)
    min_val = torch.min(text)
    max_val = torch.max(text)

    # Apply the normalization formula
    text2 = ((text - min_val) / (max_val - min_val)) * 256

    text2=text2.long()
    print(text2.shape)
    text2=text2.cuda()
   

    model(text2, return_loss = return_loss)

def test_modality_only():

    model = Transfusion(
        num_text_tokens = 256,
       #dim_latent = (384, 192),
        dim_latent = (384, 3), #second dimension in images dimension below is the second one here
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = False
        )
    )

    #images = randn(2, 192, 8, 8)
    images = randn(2, 3, 128, 128)

    loss = model(images, return_loss = True, modality_type = 1)

    loss.backward()


@pytest.mark.parametrize('custom_time_fn', (False, True))
def test_text_image_end_to_end(
    custom_time_fn: bool
):
    mock_vae_encoder = nn.Conv2d(3, 384, 3, padding = 1)
    mock_vae_decoder = nn.Conv2d(384, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 4,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = ((4, 4)),
        modality_encoder = mock_vae_encoder,
        modality_decoder = mock_vae_decoder,
        transformer = dict(
            dim = 512,
            depth = 8
        )
    )

    text_and_images = [
        [
            randint(0, 4, (16,)),
            randn(3, 8, 8),
            randint(0, 4, (8,)),
            randn(3, 7, 7)
        ],
        [
            randint(0, 4, (16,)),
            randn(3, 8, 5),
            randint(0, 4, (5,)),
            randn(3, 2, 16),
            randint(0, 4, (9,))
        ]
    ]

    # allow researchers to experiment with different time distributions across multiple modalities in a sample

    def modality_length_to_times(modality_length):
        has_modality = modality_length > 0
        return torch.where(has_modality, torch.ones_like(modality_length), 0.)

    time_fn = modality_length_to_times if custom_time_fn else None

    # forward

    loss = model(
        text_and_images,
        modality_length_to_times_fn = time_fn
    )

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample()

def test_velocity_consistency():
    mock_encoder = nn.Conv2d(3, 384, 3, padding = 1)
    mock_decoder = nn.Conv2d(384, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = ((4, 4)),
        modality_validate_num_dim = 2,
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(
            dim = 512,
            depth = 1
        )
    )

    ema_model = deepcopy(model)

    text_and_images = [
        [
            randint(0, 12, (16,)),
            randn(3, 8, 8),
            randint(0, 12, (8,)),
            randn(3, 7, 7)
        ],
        [
            randint(0, 12, (16,)),
            randn(3, 8, 5),
            randint(0, 12, (5,)),
            randn(3, 2, 16),
            randint(0, 12, (9,))
        ]
    ]

    def modality_length_to_times(modality_length):
        has_modality = modality_length > 0
        return torch.where(has_modality, torch.ones_like(modality_length), 0.)

    loss, breakdown = model(
        text_and_images,
        velocity_consistency_ema_model = ema_model,
        modality_length_to_times_fn = modality_length_to_times,
        return_breakdown = True
    )

    loss.backward()

    assert exists(breakdown.velocity)
