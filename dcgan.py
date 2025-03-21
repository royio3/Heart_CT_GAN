"""
The modified Code is based on:
*Title: PyTorch-GAN/DCGAN
*Author: Erik Linder-Norén
*Date: 2018
*Availability: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
"""
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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):

        self.img_shape = img_shape
        super(Generator, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
    
def run_dcgan(opt):
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

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

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
            valid = torch.full((imgs.shape[0], 1), 1.0, device='cuda' if cuda else 'cpu', requires_grad=False)
            fake = torch.full((imgs.shape[0], 1), 0.0, device='cuda' if cuda else 'cpu', requires_grad=False)



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
    os.makedirs("saved_models", exist_ok=True)
    it = 1
    while(os.path.exists(f"./saved_models/DCGANexpmodel{it}.pth")):
        it+=1
    torch.save(generator.state_dict(), f"./saved_models/DCGANexpmodel{it}.pth")
    print(f"model saved as: DCGANexpmodel{it}.pth in saved_models")
    return dis_loss, gen_loss

def experiment(opt):
    print("running experiments...")
    os.makedirs("plots", exist_ok=True)

    #First the default configuration is run: lr = 0.0002, latent dim = 100, batchsize = 64
    print("running default configuration : lr = 0.0002, latent dim = 100, batchsize = 64")
    opt.lr = 0.0002
    opt.latent_dim = 100
    opt.batch_size = 64
    def_dis_loss, def_gen_loss = run_dcgan(opt) #these results will be used in multiple plots

    #learning rate experiment
    print("running learning rate experiment")
    fig_dis, ax_dis = plt.subplots()    
    ax_dis.set_xlabel('Epoch') 
    ax_dis.set_ylabel('Discriminator Loss')
    ax_dis.set_title("Average discriminator loss per epoch DCGAN using different lr")

    fig_gen, ax_gen = plt.subplots()
    ax_gen.set_xlabel('Epoch') 
    ax_gen.set_ylabel('Generator Loss')
    ax_gen.set_title("Average generator loss per epoch DCGAN using different lr")

    ax_dis.plot(def_dis_loss,label=f"lr = {opt.lr}")
    ax_gen.plot(def_gen_loss,label=f"lr = {opt.lr}")
    
    lrates = [0.0001, 0.0004]

    for lrate in lrates:
        print(f"learning rate: {lrate}")
        opt.lr = lrate
        dis_loss, gen_loss = run_dcgan(opt)
        ax_dis.plot(dis_loss,label=f"lr = {opt.lr}")
        ax_gen.plot(gen_loss,label=f"lr = {opt.lr}")
    
    ax_dis.legend()
    ax_gen.legend()
    fig_dis.savefig("./plots/DCGANexp_dis_lr.png",dpi=300)
    fig_gen.savefig("./plots/DCGANexp_gen_lr.png",dpi=300)
    print(f"plots saved as: DCGANexp_dis_lr.png and DCGANexp_gen_lr.png in plots")

    opt.lr = 0.0002 #resets learning ratedo default configuration

    #latent dim experiment
    print("running latent space dimension experiment")
    fig_dis, ax_dis = plt.subplots()    
    ax_dis.set_xlabel('Epoch') 
    ax_dis.set_ylabel('Discriminator Loss')
    ax_dis.set_title("Average discriminator loss per epoch DCGAN using different latent dim")

    fig_gen, ax_gen = plt.subplots()
    ax_gen.set_xlabel('Epoch') 
    ax_gen.set_ylabel('Generator Loss')
    ax_gen.set_title("Average generator loss per epoch DCGAN using different latent dim")

    ax_dis.plot(def_dis_loss,label=f"latent dim = {opt.latent_dim}")
    ax_gen.plot(def_gen_loss,label=f"latent dim = {opt.latent_dim}")
    
    ldims = [50, 200]

    for ldim in ldims:
        print(f"latent space dimension: {ldim}")
        opt.latent_dim = ldim
        dis_loss, gen_loss = run_dcgan(opt)
        ax_dis.plot(dis_loss,label=f"latent dim = {opt.latent_dim}")
        ax_gen.plot(gen_loss,label=f"latent dim = {opt.latent_dim}")
    
    ax_dis.legend()
    ax_gen.legend()
    fig_dis.savefig("./plots/DCGANexp_dis_ldim.png",dpi=300)
    fig_gen.savefig("./plots/DCGANexp_gen_ldim.png",dpi=300)
    print(f"plots saved as: DCGANexp_dis_ldim.png and DCGANexp_gen_ldim.png in plots")

    opt.latent_dim = 100 # resets latent dim to default configuration

    #batch size experiment
    print("running batch size experiment")
    fig_dis, ax_dis = plt.subplots()    
    ax_dis.set_xlabel('Epoch') 
    ax_dis.set_ylabel('Discriminator Loss')
    ax_dis.set_title("Average discriminator loss per epoch DCGAN using different batch sizes")

    fig_gen, ax_gen = plt.subplots()
    ax_gen.set_xlabel('Epoch') 
    ax_gen.set_ylabel('Generator Loss')
    ax_gen.set_title("Average generator loss per epoch DCGAN using different batch sizes")

    ax_dis.plot(def_dis_loss,label=f"batch size = {opt.batch_size}")
    ax_gen.plot(def_gen_loss,label=f"batch size = {opt.batch_size}")
    
    batchsizes = [16, 32]

    for batchsize in batchsizes:
        print(f"batch size: {batchsize}")
        opt.batch_size = batchsize
        dis_loss, gen_loss = run_dcgan(opt)
        ax_dis.plot(dis_loss,label=f"batch size = {opt.batch_size}")
        ax_gen.plot(gen_loss,label=f"batch size = {opt.batch_size}")
    
    ax_dis.legend()
    ax_gen.legend()
    fig_dis.savefig("./plots/DCGANexp_dis_bsize.png",dpi=300)
    fig_gen.savefig("./plots/DCGANexp_gen_bsize.png",dpi=300)
    print(f"plots saved as: DCGANexp_dis_bsize.png and DCGANexp_gen_bsize.png in plots")

    print("DCGAN experiments completed!")

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--experiment", type=bool, default=False, help="Will the thesis experiments be ran?")
    opt = parser.parse_args()
    print(opt)

    if opt.experiment:
        experiment(opt)
    
    else:
        run_dcgan(opt)
