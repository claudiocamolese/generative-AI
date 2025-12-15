import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# Conditional GAN
# ----------------------------------------

class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes, generator_size, latent_size):
        super().__init__()
        self.num_classes = num_classes
        self.generator_size = generator_size
        self.latent_size = latent_size
        
        self.latent_embedding = nn.Sequential(
            nn.Linear(self.latent_size, self.generator_size // 2),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.num_classes, self.generator_size // 2),
        )
        self.tcnn = nn.Sequential(
        nn.ConvTranspose2d( self.generator_size, self.generator_size, 4, 1, 0),
        nn.BatchNorm2d(self.generator_size),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d( self.generator_size, self.generator_size // 2, 3, 2, 1),
        nn.BatchNorm2d(self.generator_size // 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d( self.generator_size // 2, self.generator_size // 4, 4, 2, 1),
        nn.BatchNorm2d(self.generator_size // 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d( self.generator_size // 4, 1, 4, 2, 1),
        nn.Tanh()
        )
        
    def forward(self, latent, condition):
        vec_latent = self.latent_embedding(latent)
        vec_class = self.condition_embedding(condition)
        combined = torch.cat([vec_latent, vec_class], dim=1).reshape(-1, self.generator_size, 1, 1)
        return self.tcnn(combined)
            
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes, discriminator_size, hidden_size):
        super().__init__()
        self.num_classes = num_classes
        self.discriminator_size = discriminator_size
        self.hidden_size = hidden_size
        
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.num_classes, self.discriminator_size * 4),
        )
        self.cnn_net = nn.Sequential(
        nn.Conv2d(1, self.discriminator_size // 4, 3, 2),
        nn.InstanceNorm2d(self.discriminator_size // 4, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(self.discriminator_size // 4, self.discriminator_size // 2, 3, 2),
        nn.InstanceNorm2d(self.discriminator_size // 2, affine=True),
        nn.LeakyReLU(0.2, inplace=True),   
        nn.Conv2d(self.discriminator_size // 2, self.discriminator_size, 3, 2),
        nn.InstanceNorm2d(self.discriminator_size, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Flatten(),
        )
        self.Discriminator_net = nn.Sequential(
        nn.Linear(self.discriminator_size * 8, self.hidden_size),
        nn.LeakyReLU(0.2, inplace=True),   
        nn.Linear(self.hidden_size, 1),
        )
        
    def forward(self, image, condition):
        vec_condition = self.condition_embedding(condition)
        cnn_features = self.cnn_net(image)
        combined = torch.cat([cnn_features, vec_condition], dim=1)
        return self.Discriminator_net(combined)


def weights_init_gan(m):
    """
    Standard weight initialization for GAN/WGAN:
    Normal distribution with 0 mean and 0.02 standard deviation
    """
    classname = m.__class__.__name__
    
    # 1. Liniar and Convolutional layers
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    
    # 2. Normalization layers
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)