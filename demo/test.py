from models import Generator, Discriminator
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os
import cv2
from time import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

if __name__ == '__main__':
    device = "cuda:0"
    generator = Generator(128).to(device)
    generator.load_state_dict(torch.load(str(ROOT / 'generator_1690944447.3646827.pth')))
    generator.eval()
    discriminator = Discriminator(128).to(device)
    discriminator.load_state_dict(torch.load(str(ROOT / 'discriminator_1690944447.3646827.pth')))
    discriminator.eval()
    gen = torch.Generator(device=device)
    
    for i in range(1000):
        gen.manual_seed(int(time()))
        random_noise = torch.rand(size=(1,100), generator=gen, device=device)
        generated_image = generator(random_noise)
        # fig = plt.figure(figsize=(8, 8))
        # fig.add_subplot(1, 3, 1)

        generated_image = generated_image.cpu().detach().numpy() * 255
        generated_image = generated_image.astype(np.uint8)[0].transpose(1, 2, 0)
        # plt.imshow(generated_image)
        # plt.show()
        cv2.imshow('src',generated_image)
        cv2.waitKey()
        
        