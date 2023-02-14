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
from dcgan import DCGenerator_simple, DCDiscriminator_simple
from utils import weights_init

## ________ Setting parameters ________
QUICK_DESCRIPTION_OF_EXPERIMENT = """WGAN with gradient penalty: n_disc = 3, 2 extra layers
SUBSET with only one class of CIFAR10 and modified training function. """

n_epochs = 10
display_step = 500          # Frequency for saving images
clip_value = None           # if None: no clipping -> gradient penalty instead
n_disc = 5                  # How many more times the discriminator (=critic) is trained compared to the generator? (-> see WGAN paper)
optimizer = 'Adam' #'RMSProp'
extra_layers = 0            # Extra layer in both architectures

## Optionally load from previous checkpoints 
checkpoint_disc = None # Does not load anything and starts from the beginning
checkpoint_gen = None
# checkpoint_disc = "" #Loads from the defined file
# checkpoint_gen = ""

description = f"""///{QUICK_DESCRIPTION_OF_EXPERIMENT}///
{n_epochs} epochs. {display_step} display_step. {optimizer} optimizer. {n_disc} n_disc.
standard WGAN with clipping gradients. clip_value = {clip_value}
checkpoint_disc ? {checkpoint_disc}. checkpoint_gen ? {checkpoint_gen}.
"""

file_path = now.strftime("checkpoints/test_%Y-%m-%d_%H:%M:%S")
os.mkdir(file_path+"/")

with open(file_path+"/README.txt", 'w', ) as f:
    f.write(description)
print(description)

## ________ Setting parameters ________
criterion = nn.BCEWithLogitsLoss()
z_dim = 64
batch_size = 128
lr = 0.0002 # Learning rate 

# These parameters control the optimizer's momentum, which you can read more about here: https://distill.pub/2017/momentum/
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Create a dataloader from MNIST 
MNIST_data = MNIST(root='.', train=True, download=True, transform=transform)
dataloader = DataLoader(MNIST_data, 
                        batch_size=batch_size, 
                        shuffle=True)

## ________ Initialization ________
gen = DCGenerator_simple(z_dim=z_dim).to(device)
disc = DCDiscriminator_simple().to(device)

if checkpoint_disc is not None:
    disc.load_state_dict(torch.load(checkpoint_disc))
    gen.load_state_dict(torch.load(checkpoint_gen))

if optimizer == 'Adam':
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