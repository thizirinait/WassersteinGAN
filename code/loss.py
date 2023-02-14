import torch
from utils import get_noise


## Loss functions for a normal GAN
def get_disc_loss_GAN(gen, disc, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    generated = gen(noise).detach()
    pred_generated = disc(generated)
    loss_generated = criterion(pred_generated, torch.zeros(num_images, 1).to(device))
    
    pred_real = disc(real)
    loss_real = criterion(pred_real, torch.ones(num_images, 1).to(device))
    disc_loss = (loss_generated + loss_real) / 2
    return disc_loss

def get_gen_loss_GAN(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    generated = gen(noise)
    pred_generated = disc(generated)
    gen_loss = criterion(pred_generated, torch.ones(num_images,1).to(device))
    return gen_loss


## Loss functions for a WGAN
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    '''Calculates the gradient penalty loss for WGAN'''
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = D(interpolates).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs= torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_disc_loss_WGAN(gen, disc, real, num_images, z_dim, device, lambd=None):
    """
    if lambd is None: no gradient penalty -> do weigth clipping
    """
    noise = get_noise(num_images, z_dim, device=device)
    with torch.no_grad():
        fake = gen(noise)
    fake_output = disc(fake).mean()

    real_output = - disc(real).mean() ## signed inverted according to the pseudo-code
    output = fake_output + real_output

    if lambd is not None:
        output += lambd * compute_gradient_penalty(disc, real, fake, device)
    return output

def get_gen_loss_WGAN(gen, disc, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    fake_output = - disc(fake).mean()
    return fake_output