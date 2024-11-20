import argparse
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import os
from wgan import Generator 

cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument("--modelname", type=str,  help="filename of saved model")
parser.add_argument("--generate_img", type=int, default=10,  help="How many images need to be generated")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--image_name", type=str, default="", help="name image file")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
if not os.path.exists(f"./saved_models/{opt.modelname}"):
    print("Error model does not exist in saved_model")
    exit(-1)

generator = Generator(img_shape, opt.latent_dim)

if cuda:
    generator.cuda()

generator.load_state_dict(torch.load(f"./saved_models/{opt.modelname}" ,weights_only=False))

generator.eval() 

os.makedirs("./generated_images", exist_ok=True)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

num_generated_img_per_img = 16
if opt.image_name == "":
    num_generated_img_per_img = 1
for i in range(opt.generate_img):
    z = torch.tensor(np.random.normal(0, 1, (num_generated_img_per_img, opt.latent_dim)), dtype=torch.float32, device='cuda' if cuda else 'cpu')


    gen_img = generator(z)
    #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    if opt.image_name == "":
        save_image(gen_img.data, f"./generated_images/fake_image_{i}.png", normalize=True)
    else:
        save_image(gen_img.data[:16], f"./generated_images/{opt.image_name}", nrow=4, normalize=True)
    print(f"Generated image {i+1}/{opt.generate_img}")