import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录，这里为mycompany
sys.path.append(BASE_DIR)
import torch
from models.gan import Generator
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from utils.options import args_parser
def Augment():
    args = args_parser()
    G = Generator(args=args).cuda()
    img_list = []
    for i in range(10):
        G.load_state_dict(torch.load('./testoutput/generator'+str(i)+'.pth'))
        noise = torch.randn(64, 100, 1, 1).cuda()
        fake = G(noise).detach().cpu()
        img_list.append(vutils.make_grid(fake.detach().cpu(), padding=2, normalize = True))
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

if __name__ == '__main__':
    Augment()
    


