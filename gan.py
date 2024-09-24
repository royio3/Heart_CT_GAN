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
    



def run_gan(opt):
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    #Change image in to grayscale image
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

    # Initialize generator and discriminator
    generator = Generator(img_shape, opt.latent_dim)
    discriminator = Discriminator(img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()



  # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

  # ----------
  #  Training
  # ----------

    gen_loss = [0] *opt.n_epochs
    dis_loss = [0] *opt.n_epochs

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

            dis_loss[epoch] += d_loss.item()
            gen_loss[epoch] += g_loss.item()

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        print(
            "[Epoch %d/%d] [average D loss: %f] [average G loss: %f]"
            % (epoch, opt.n_epochs, dis_loss[epoch]/len(dataloader), gen_loss[epoch]/len(dataloader))
        )
    for i in range(opt.n_epochs):
        dis_loss[i] = dis_loss[i]/len(dataloader)
        gen_loss[i] = gen_loss[i]/len(dataloader)

    it = 1
    os.makedirs("saved_models", exist_ok=True)
    while(os.path.exists(f"./saved_models/GANexpmodel{it}.pth")):
        it+=1
    torch.save(generator.state_dict(), f"./saved_models/GANexpmodel{it}.pth")
    print(f"model saved as: GANexpmodel{it}.pth in saved_models")

    return dis_loss, gen_loss


def experiment(opt):
    print("running experiments...")
    os.makedirs("plots", exist_ok=True)

    #First the default configuration is run: lr = 0.0002, latent dim = 100, batchsize = 64
    print("running default configuration : lr = 0.0002, latent dim = 100, batchsize = 64")
    opt.lr = 0.0002
    opt.latent_dim = 100
    opt.batch_size = 64
    def_dis_loss, def_gen_loss = run_gan(opt) #these results will be used in multiple plots

    #learning rate experiment
    print("running learning rate experiment")
    fig_dis, ax_dis = plt.subplots()    
    ax_dis.set_xlabel('Epoch') 
    ax_dis.set_ylabel('Discriminator Loss')
    ax_dis.set_title("Average discriminator loss per epoch GAN using different lr")

    fig_gen, ax_gen = plt.subplots()
    ax_gen.set_xlabel('Epoch') 
    ax_gen.set_ylabel('Generator Loss')
    ax_gen.set_title("Average generator loss per epoch GAN using different lr")

    ax_dis.plot(def_dis_loss,label=f"lr = {opt.lr}")
    ax_gen.plot(def_gen_loss,label=f"lr = {opt.lr}")
    
    lrates = [0.0001, 0.0004]

    for lrate in lrates:
        print(f"learning rate: {lrate}")
        opt.lr = lrate
        dis_loss, gen_loss = run_gan(opt)
        ax_dis.plot(dis_loss,label=f"lr = {opt.lr}")
        ax_gen.plot(gen_loss,label=f"lr = {opt.lr}")
    
    ax_dis.legend()
    ax_gen.legend()
    fig_dis.savefig("./plots/GANexp_dis_lr.png",dpi=300)
    fig_gen.savefig("./plots/GANexp_gen_lr.png",dpi=300)
    print(f"plots saved as: GANexp_dis_lr.png and GANexp_gen_lr.png in plots")

    opt.lr = 0.0002 #resets learning ratedo default configuration

    #latent dim experiment
    print("running latent space dimension experiment")
    fig_dis, ax_dis = plt.subplots()    
    ax_dis.set_xlabel('Epoch') 
    ax_dis.set_ylabel('Discriminator Loss')
    ax_dis.set_title("Average discriminator loss per epoch GAN using different latent dim")

    fig_gen, ax_gen = plt.subplots()
    ax_gen.set_xlabel('Epoch') 
    ax_gen.set_ylabel('Generator Loss')
    ax_gen.set_title("Average generator loss per epoch GAN using different latent dim")

    ax_dis.plot(def_dis_loss,label=f"latent dim = {opt.latent_dim}")
    ax_gen.plot(def_gen_loss,label=f"latent dim = {opt.latent_dim}")
    
    ldims = [50, 200]

    for ldim in ldims:
        print(f"latent space dimension: {ldim}")
        opt.latent_dim = ldim
        dis_loss, gen_loss = run_gan(opt)
        ax_dis.plot(dis_loss,label=f"latent dim = {opt.latent_dim}")
        ax_gen.plot(gen_loss,label=f"latent dim = {opt.latent_dim}")
    
    ax_dis.legend()
    ax_gen.legend()
    fig_dis.savefig("./plots/GANexp_dis_ldim.png",dpi=300)
    fig_gen.savefig("./plots/GANexp_gen_ldim.png",dpi=300)
    print(f"plots saved as: GANexp_dis_ldim.png and GANexp_gen_ldim.png in plots")

    opt.latent_dim = 100 # resets latent dim to default configuration

    #batch size experiment
    print("running batch size experiment")
    fig_dis, ax_dis = plt.subplots()    
    ax_dis.set_xlabel('Epoch') 
    ax_dis.set_ylabel('Discriminator Loss')
    ax_dis.set_title("Average discriminator loss per epoch GAN using different batch sizes")

    fig_gen, ax_gen = plt.subplots()
    ax_gen.set_xlabel('Epoch') 
    ax_gen.set_ylabel('Generator Loss')
    ax_gen.set_title("Average generator loss per epoch GAN using different batch sizes")

    ax_dis.plot(def_dis_loss,label=f"batch size = {opt.batch_size}")
    ax_gen.plot(def_gen_loss,label=f"batch size = {opt.batch_size}")
    
    batchsizes = [32, 128]

    for batchsize in batchsizes:
        print(f"latent space dimension: {batchsize}")
        opt.batch_size = batchsize
        dis_loss, gen_loss = run_gan(opt)
        ax_dis.plot(dis_loss,label=f"batch size = {opt.batch_size}")
        ax_gen.plot(gen_loss,label=f"batch size = {opt.batch_size}")
    
    ax_dis.legend()
    ax_gen.legend()
    fig_dis.savefig("./plots/GANexp_dis_bsize.png",dpi=300)
    fig_gen.savefig("./plots/GANexp_gen_bsize.png",dpi=300)
    print(f"plots saved as: GANexp_dis_bsize.png and GANexp_gen_bsize.png in plots")

    print("GAN experiments completed!")








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
    parser.add_argument("--experiment", type=bool, default=False, help="Will the thesis experiments be ran?")
    opt = parser.parse_args()
    print(opt)

    if experiment:
        experiment(opt)
    
    else:
        run_gan(opt)