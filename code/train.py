from tqdm import tqdm
from utils import show_tensor_images, save_tensor_images
from loss import *
import pickle

def train_GAN(disc, disc_opt, gen, gen_opt, z_dim, criterion, dataloader, n_epochs, display_step, file_path, device):
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    print(device)

    G_losses = []
    D_losses = []
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss_GAN(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            mean_discriminator_loss += disc_loss.item() / display_step
            disc_loss.backward()
            disc_opt.step()

            ## Update generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss_GAN(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the losses
            mean_generator_loss += gen_loss.item() / display_step
            G_losses.append(mean_generator_loss)
            D_losses.append(mean_discriminator_loss)

            ## Visualization code
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch}/{n_epochs}. Step {cur_step}. Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                # show_tensor_images(fake)
                # show_tensor_images(real)
                save_tensor_images(fake, f"{file_path}/fake_{cur_step}.png")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
        
        # Checkpoint
        torch.save(gen.state_dict(), f'{file_path}/gen_epoch_{epoch}.pth')
        torch.save(disc.state_dict(), f'{file_path}/disc_epoch_{epoch}.pth')


    with open(f"{file_path}/G_losses.pkl", "wb") as f:
        pickle.dump(G_losses, f)
    with open(f"{file_path}/D_losses.pkl", "wb") as f:
        pickle.dump(D_losses, f)

    return G_losses, D_losses

def train_WGAN(disc, disc_opt, gen, gen_opt, z_dim, dataloader, n_epochs, n_disc, display_step, file_path, device, clip_value = None):
    """
    if clip_value is None: does gradient penalty
    """

    if clip_value is None: # WGAN with Gradient Penalty
        lambda_grad_penalty = 0.01
    else: # 'Standard' WGAN
        lambda_grad_penalty = None

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    print(device)

    G_losses = []
    D_losses = []
    fake_noise_fixed = get_noise(25, z_dim, device=device)
    ind_update_generator = 0 #When to update the generator
    for epoch in range(n_epochs*n_disc): # n_epochs is multiplied by n_disc, to count the effective update of the generator
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss_WGAN(gen, disc, real, cur_batch_size, z_dim, device, lambd=lambda_grad_penalty)
            disc_loss.backward()
            disc_opt.step()
            
            if clip_value is not None: # Clip weights of discriminator if standard WGAN
                for p in disc.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            if ind_update_generator % n_disc == 0:
                ## Update generator only every n_disc iteration
                gen_opt.zero_grad()
                gen_loss = get_gen_loss_WGAN(gen, disc, cur_batch_size, z_dim, device)
                gen_loss.backward()
                gen_opt.step()
                ind_update_generator = 0 #Put back at 0 every n_disc iteration

                # Keep track of the losses every n_disc iteration
                mean_discriminator_loss += disc_loss.item() / display_step
                mean_generator_loss += gen_loss.item() / display_step
                G_losses.append(mean_generator_loss)
                D_losses.append(mean_discriminator_loss)
            ind_update_generator += 1

            ## Visualization code
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch}/{n_epochs}. Step {cur_step}. Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                with torch.no_grad():
                    fake = gen(fake_noise_fixed)
                    # show_tensor_images(fake)
                    # show_tensor_images(real)
                    save_tensor_images(fake, f"{file_path}/fake_{cur_step}.png")


                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1

        # Checkpoint
        torch.save(gen.state_dict(), f'{file_path}/gen_epoch_{epoch}.pth')
        torch.save(disc.state_dict(), f'{file_path}/disc_epoch_{epoch}.pth')

    with open(f"{file_path}/G_losses.pkl", "wb") as f:
        pickle.dump(G_losses, f)
    with open(f"{file_path}/D_losses.pkl", "wb") as f:
        pickle.dump(D_losses, f)

    return G_losses, D_losses
        