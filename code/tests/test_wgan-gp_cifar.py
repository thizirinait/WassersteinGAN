import os
from datetime import datetime
now = datetime.now()

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train import train_WGAN
from dcgan import DCGenerator, DCDiscriminator
from utils import weights_init

## ________ Setting parameters ________

QUICK_DESCRIPTION_OF_EXPERIMENT = """WGAN with gradient penalty: n_disc = 5 for the first time, and 2 extra layers (beginning slowly) 
checkpoint 18 used from previous experiment"""

n_epochs = 100
display_step = 500          # Frequency for saving images
clip_value = None           # if None: no clipping -> gradient penalty instead
n_disc = 5                  # How many more times the discriminator (=critic) is trained compared to the generator? (-> see WGAN paper)
optimizer = 'RMSProp'       #'RMSProp' #'Adam'
extra_layers = 2            # Extra layer in both architectures

## Optionally load from previous checkpoints 
# checkpoint_disc = None # Does not load anything and starts from the beginning
# checkpoints_gen = None
checkpoint_disc = "images/test_2021-01-21_21:10:26/disc_epoch_18.pth" #Loads from the defined file
checkpoint_gen = "images/test_2021-01-21_21:10:26/gen_epoch_18.pth"

## Saving information and images
file_path = now.strftime("checkpoints/test_%Y-%m-%d_%H:%M:%S")
os.mkdir(file_path+"/")

description = f"""///{QUICK_DESCRIPTION_OF_EXPERIMENT}///
{n_epochs} epochs. {display_step} display_step. {optimizer} optimizer. {n_disc} n_disc.
{extra_layers} extra_layers. clip_value ? {clip_value}.
checkpoint_disc ? {checkpoint_disc}. checkpoint_gen ? {checkpoint_gen}.
"""
with open(file_path+"/README.txt", 'w', ) as f:
    f.write(description)
print(description)

## ________ Setting parameters ________
criterion = nn.BCEWithLogitsLoss()
z_dim = 64
batch_size = 128

device = 'cuda'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Create a dataloader from CIFAR
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


## ________ Initialization ________
gen = DCGenerator(z_dim=z_dim, extra_layers=extra_layers).to(device)
disc = DCDiscriminator(extra_layers=extra_layers).to(device)

if checkpoint_disc is not None:
    disc.load_state_dict(torch.load(checkpoint_disc))
    gen.load_state_dict(torch.load(checkpoint_gen))

if optimizer == 'Adam':
    # These parameters control the optimizer's momentum, which you can read more about here: https://distill.pub/2017/momentum/
    beta_1 = 0.5 
    beta_2 = 0.999
    lr = 0.0002 # Learning rate 
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
elif optimizer == 'RMSProp':
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=0.0005)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=0.0005)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

print(gen)
print(disc)

## ________ Training ________
G_losses, D_losses = train_WGAN(disc, disc_opt, gen, gen_opt, z_dim, dataloader, n_epochs, n_disc, display_step, file_path, device, clip_value = clip_value)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator")
plt.plot(D_losses,label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()