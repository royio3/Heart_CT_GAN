"""
Small parts of the code are based of on
*Title: PyTorch-GAN
*Author: Erik Linder-Norén
*Date: 2018
*Availability: https://github.com/eriklindernoren/PyTorch-GAN/
"""
import argparse
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import os

cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument("--modelname", type=str,  help="filename of saved model")
parser.add_argument("--generate_img", type=int, default=10,  help="How many images need to be generated")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--image_name", type=str, default="", help="name image file")
parser.add_argument("--group", type=bool, default=False, help="Store as group 4x4 group image")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
if not os.path.exists(f"./saved_models/{opt.modelname}"):
    print("Error model does not exist in saved_model")
    exit(-1)
if opt.modelname.startswith("DCGAN"):
    from dcgan import Generator
    print("DCGAN model found")

elif opt.modelname.startswith("WGAN"):
    from wgan import Generator
    print("WGAN model found")
else:
    from gan import Generator
    print("GAN model found")

generator = Generator(img_shape, opt.latent_dim)

if cuda:
    generator.cuda()

generator.load_state_dict(torch.load(f"./saved_models/{opt.modelname}" ,weights_only=False))

generator.eval() 

os.makedirs("./generated_images", exist_ok=True)
os.makedirs(f"./generated_images/{opt.modelname}_images", exist_ok=True)



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#For non group images
if opt.group == False:
    num_generated_img_per_img = 1
    for i in range(opt.generate_img):
        z = torch.tensor(np.random.normal(0, 1, (num_generated_img_per_img, opt.latent_dim)), dtype=torch.float32, device='cuda' if cuda else 'cpu')
        gen_img = generator(z)
        if opt.image_name == "":
            save_image(gen_img.data, f"./generated_images/{opt.modelname}_images/fake_image_{i}.png", normalize=True)
            print(f"Generated image {i+1}/{opt.generate_img}")
        else:
            save_image(gen_img.data, f"./generated_images/{opt.modelname}_images/{opt.image_name}_{i}.png", normalize=True)
            print(f"Generated image {i+1}/{opt.generate_img}")

#For 4x4 group image
else:
    num_generated_img_per_img = 16
    z = torch.tensor(np.random.normal(0, 1, (num_generated_img_per_img, opt.latent_dim)), dtype=torch.float32, device='cuda' if cuda else 'cpu')
    gen_img = generator(z)
    if opt.image_name == "":
        save_image(gen_img.data[:16], f"./generated_images/{opt.modelname}_images/{opt.image_name}", nrow=4, normalize=True)
        print(f"group image saved in ./generated_images/{opt.modelname}_images/{opt.image_name}")

    save_image(gen_img.data[:16], f"./generated_images/{opt.modelname}_images/{opt.image_name}", nrow=4, normalize=True)
    print(f"group image saved in ./generated_images/{opt.modelname}_images/{opt.image_name}")