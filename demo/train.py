import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import sys
import os
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models import Generator, Discriminator
from time import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

if __name__ == "__main__":
    cls_num = 1
    batch_size =  8
    epochs = 1000
    device = "cuda:0"
    # device = "cpu"
    generator = Generator(128).to(device)
    discriminator = Discriminator(128).to(device)
    # discriminator.load_state_dict(torch.load(str(ROOT / 'discriminator.pth')))
    loss_fn = nn.BCEWithLogitsLoss()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=1e-4)
    generator_optimizer = torch.optim.Adam(generator.parameters(),lr=1e-4)

    transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    training_data = datasets.ImageFolder(
        str(ROOT / "data/tmp/"), transform=transform
    )
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    gen = torch.Generator(device=device)
    start_t = str(time())
    for t in range(epochs):
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar)  # progress bar
        print(f"Epoch {t+1}\n-------------------------------")
        for batch, (x, y) in pbar:
            true_data = x.to(device)
            true_labels = torch.ones((batch_size), device=device).view(-1)

            gen.manual_seed(int(time()))
            noise = torch.rand(size=(batch_size,100), generator=gen, device=device)
            generated_data = generator(noise)

            generator_optimizer.zero_grad()
            generator_discriminator_out = discriminator(generated_data)
            generator_loss = loss_fn(generator_discriminator_out.view(-1), true_labels)
            generator_loss.backward()
            generator_optimizer.step()


            discriminator_optimizer.zero_grad()
            true_discriminator_out = discriminator(true_data)
            true_discriminator_loss = loss_fn(true_discriminator_out.view(-1), true_labels)

            generator_discriminator_out = discriminator(generated_data.detach())
            generator_discriminator_loss = loss_fn(
                generator_discriminator_out.view(-1), torch.zeros(batch_size).to(device).view(-1)
            )
            discriminator_loss = (
                true_discriminator_loss + generator_discriminator_loss
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()
            if batch % 5 == 0:
                loss1,loss2, current = generator_loss.item(), discriminator_loss.item(), (batch + 1) * len(x)
                print(f"loss1: {loss1:>7f} loss2: {loss2:>7f}  [{current:>5d}]")
        
        torch.save(discriminator.state_dict(), str(ROOT / 'discriminator_{}.pth'.format(start_t)))
        torch.save(generator.state_dict(), str(ROOT / 'generator_{}.pth'.format(start_t)))
