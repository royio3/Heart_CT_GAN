#Code based on eriklindernoren "PyTorch-GAN" https://github.com/eriklindernoren/PyTorch-GAN
# Roy timman 2024
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate") #0.0002
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(img_shape, opt.latent_dim)
    discriminator = Discriminator(img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    if opt.channels == 1:    
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
          
    else:
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]) 
    
  # Configure data loader
    dataset = datasets.ImageFolder(root='./dataset/', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

  # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

  # ----------
  #  Training
  # ----------

    gen_loss = [0] *opt.n_epochs
    dis_loss = [0] *opt.n_epochs

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch') 
    ax.set_ylabel('Loss')
    ax.set_title("Discriminator and generator loss per epoch GAN")
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

          # Adversarial ground truths
            valid = torch.full((imgs.size(0), 1), 1.0, dtype=torch.float32, device='cuda' if cuda else 'cpu', requires_grad=False)
            fake = torch.full((imgs.size(0), 1), 0.0, dtype=torch.float32, device='cuda' if cuda else 'cpu', requires_grad=False)


            # Configure input
            real_imgs = imgs.to(dtype=torch.float32, device='cuda' if cuda else 'cpu')

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.size(0), opt.latent_dim, device='cuda' if cuda else 'cpu')


            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            dis_loss[epoch] += d_loss.item()
            gen_loss[epoch] += g_loss.item()

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
  
    for i in range(opt.n_epochs):
        dis_loss[i] = dis_loss[i]/len(dataloader)
        gen_loss[i] = gen_loss[i]/len(dataloader)
    ax.plot(dis_loss,label=f"Average Discriminator Loss per epoch")
    ax.plot(gen_loss,label=f"Average Generator Loss per epoch")
    ax.legend()
    it = 1
    os.makedirs("plots", exist_ok=True)
    while(os.path.exists(f"./plots/GANexp{it}.png")):
        it+=1
    fig.savefig(f"./plots/GANexp{it}.png",dpi=300)
    print(f"plot saved as: GANexp{it}.png in plots")

    it = 1
    os.makedirs("saved_models", exist_ok=True)
    while(os.path.exists(f"./saved_models/GANexpmodel{it}.pth")):
        it+=1
    torch.save(generator.state_dict(), f"./saved_models/GANexpmodel{it}.pth")
    print(f"model saved as: GANexpmodel{it}.pth in saved_models")