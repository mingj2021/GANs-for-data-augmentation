import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Discriminator(nn.Module):
    def __init__(self, discriminator_depth) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=discriminator_depth,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=discriminator_depth,
                out_channels=discriminator_depth * 2,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(discriminator_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=discriminator_depth * 2,
                out_channels=discriminator_depth * 4,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(discriminator_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=discriminator_depth * 4,
                out_channels=discriminator_depth * 8,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(discriminator_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=discriminator_depth * 8,
                out_channels=1,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        return x


class Generator(nn.Module):
    def __init__(self, generator_depth) -> None:
        super().__init__()
        self.generator_depth = generator_depth
        self.layer1 = nn.Sequential(
            nn.Linear(100, 8 * 8 * generator_depth),
            nn.BatchNorm1d(num_features=8 * 8 * generator_depth),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=generator_depth,
                out_channels=generator_depth * 8,
                kernel_size=(4, 4),
                stride=1,
                padding=3,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=generator_depth * 8),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=generator_depth * 8,
                out_channels=generator_depth * 4,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=generator_depth * 4),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=generator_depth * 4,
                out_channels=generator_depth * 2,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=generator_depth * 2),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=generator_depth * 2,
                out_channels=generator_depth * 1,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=generator_depth * 1),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=generator_depth,
                out_channels=3,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                # dilation=2,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.layer1(x)
        # print(x.shape)
        x = x.reshape(-1, self.generator_depth, 8, 8)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        x = self.output_layer(x)
        # print(x.shape)
        return x


if __name__ == "__main__":
    generator = Generator(64)
    discriminator = Discriminator(128)

    random_noise = torch.randn(8, 100)
    generated_image = generator(random_noise)
    print(generated_image.shape)
    print('--------------------------------')
    pred = discriminator(generated_image)
    print(pred.shape)
    
