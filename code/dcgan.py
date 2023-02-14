
import torch
import torchvision
from torch import nn

# Basic architectures used in TP2, they are not used anymore (for now)
class DCGenerator_simple(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(DCGenerator_simple, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim*4, kernel_size=3, stride=2),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=3, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2),
            nn.Tanh()
        )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

class DCDiscriminator_simple(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(DCDiscriminator_simple, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(hidden_dim*2, 1, kernel_size=4, stride=2)                        
        )
    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

# More complex architectures: possibility to add extra layers
class DCGenerator(nn.Module):
    def __init__(self, z_dim=10, im_chan=3, im_size_power=5, hidden_dim=64, extra_layers = 0):
        super(DCGenerator, self).__init__()
        self.z_dim = z_dim
        #Nota bene: im_size must be a power of 2: im_size = 2 ** im_size_power = 2**5 = 32 by default
        
        hidden_dim = hidden_dim * 2**(im_size_power-3)

        ## _____ Start _____
            # size 1 -> 4
        gen = nn.Sequential()
        gen.add_module(f'start:{z_dim}-{hidden_dim}:ConvT',
                       nn.ConvTranspose2d(z_dim, hidden_dim, kernel_size=4, stride=1))
        gen.add_module(f'start:BN',
                       nn.BatchNorm2d(hidden_dim))
        gen.add_module(f'start:ReLU',
                       nn.ReLU(inplace=True))
        
        ## _____ Middle layers _____
        for i in range(im_size_power - 3): # size *= 2 at each iteration, from 4 to 2**(im_size_power-1)
            gen.add_module(f'>{i}:{hidden_dim}-{hidden_dim // 2}:ConvT',
                        nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1))
            gen.add_module(f'>{i}:BN',
                          nn.BatchNorm2d(hidden_dim // 2))
            gen.add_module(f'>{i}:ReLU',
                          nn.ReLU(inplace=True))
            hidden_dim = hidden_dim // 2
        
        ## _____ Extra layers _____
        for i in range(extra_layers): # size = size at each iteration
            gen.add_module(f'+extra {i}:{hidden_dim}-{hidden_dim}:ConvT',
                          nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            gen.add_module(f'+extra {i}:BN',
                          nn.BatchNorm2d(hidden_dim))
            gen.add_module(f'+extra {i}:ReLU',
                          nn.ReLU(inplace=True))

        ## _____ Final layers _____
        gen.add_module(f'end:{hidden_dim}-{im_chan}:ConvT',
                       nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1))
        gen.add_module(f'end:Tanh',
                       nn.Tanh())

        self.gen = gen

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

class DCDiscriminator(nn.Module):
    def __init__(self, im_chan=3, im_size_power=5, hidden_dim=64, extra_layers=0):
        super(DCDiscriminator, self).__init__()

        ## _____ Start _____
            # size 2**im_size_power -> 2**(im_size_power-1)
        disc = nn.Sequential()
        disc.add_module(f'start:{im_chan}-{hidden_dim}:Conv',
                        nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1))
        disc.add_module(f'start:LeakyReLU',
                        nn.LeakyReLU(0.2, inplace=True))

        ## _____ Extra layers _____
        for i in range(extra_layers): #constant size
            disc.add_module(f'+extra {i}:{hidden_dim}-{hidden_dim}:Conv',
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            disc.add_module(f'+extra {i}:BN',
                            nn.BatchNorm2d(hidden_dim))
            disc.add_module(f'+extra {i}:LeakyReLU',
                            nn.LeakyReLU(0.2, inplace=True))
            
        ## _____ Middle layers _____
        for i in range(im_size_power - 3): # size /= 2 from 2**(im_size_power-1) to 4
            disc.add_module(f'>{i}:{hidden_dim}-{hidden_dim*2}:Conv',
                            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1))
            disc.add_module(f'>{i}:BN',
                            nn.BatchNorm2d(hidden_dim*2))
            disc.add_module(f'>{i}:LeakyReLU',
                            nn.LeakyReLU(0.2, inplace=True))
            hidden_dim *= 2

        ## _____ Final layers _____ size from 4 to 1
        disc.add_module(f'end:{hidden_dim}-1:Conv',
                        nn.Conv2d(hidden_dim, 1, 4, 1, 0, bias=False))

        self.disc = disc

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
