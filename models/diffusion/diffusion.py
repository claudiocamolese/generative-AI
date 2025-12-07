import torch
import torch.nn as nn
import os

from tqdm import tqdm
from torchviz import make_dot

from .utils.gaussian_projection import GaussianFourierProjection
from .utils.fully_connected import FullyConnected
from .spatial_transformer import SpatialTransformer
from .utils.probability import diffusion_coeff, marginal_prob_std

class DiffusionModel(nn.Module):
    """
        This class is a U-Net architecture enhanced with Spatial Transformer blocks for class-conditioned
    diffusion models on image data (e.g., MNIST, Fashion MNIST). The network combines convolutional 
    downsampling/upsampling with global self- and cross-attention layers to better capture 
    long-range dependencies in spatial feature maps.

    The model predicts the score/noise ε(x_t, t, y) required by denoising diffusion 
    probabilistic models (DDPM/SDE-based models). Conditioning is performed via:
    - Continuous time embedding using Gaussian Fourier features.
    - Discrete class embedding for classifier-free or class-guided diffusion.

    Architecture:
        - Four convolutional encoder blocks with time conditioning.
        - Two SpatialTransformer blocks at the deeper layers for global attention.
        - Three transposed-convolution decoder blocks with skip connections.
        - Final 1-channel reconstruction layer.
        - Output is normalized by the diffusion marginal σ(t).
    
    Returns:
        Tensor:
            Predicted noise ε_θ(x_t, t, y) of shape (B, 1, H, W).

    """
    def __init__(self, 
                 marginal_prob_std = marginal_prob_std, 
                 channels= [32, 64, 128, 256], 
                 embed_dim= 256,
                 text_dim= 256, 
                 class_num= 10):
        """
        Args:
            marginal_prob_std (_type_): function mapping diffusion time t → σ(t), the marginal standard deviation
            of the perturbation kernel. used to normalize the predicted noise.
            channels (list, optional): number of convolutional feature channels at each U-Net resolution level. Defaults to [32, 64, 128, 256].
            embed_dim (int, optional): dimensionality of the continuous time embedding. Defaults to 256.
            text_dim (int, optional): dimensionality of the class embedding used as context in transformer blocks. Defaults to 256.
            class_num (int, optional): number of classes in dataset. Defaults to 10.
        """
        
        super().__init__()
        # continuos time embedding for diffusion 
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim= embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )# [batch, embed_dim]

        # generate embedding for the mnist classes (0-9) to incorporate in the conditional diffusion  
        self.cond_embed = nn.Embedding(class_num, text_dim)

        # Other model properties
        self.gelu = nn.GELU()
        self.marginal_prob_std = marginal_prob_std
        
        """------------------------
            ENCODING BLOCK
        ------------------------"""
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride= 1, bias= False)
        self.dense1 = FullyConnected(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride= 2, bias= False)
        self.dense2 = FullyConnected(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride= 2, bias= False)
        self.dense3 = FullyConnected(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride= 2, bias= False)
        self.dense4 = FullyConnected(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)

        """---------------------
            DECODING BLOCK
        ----------------------"""
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride= 2, bias= False)
        self.dense5 = FullyConnected(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride= 2, bias= False, output_padding= 1)
        self.dense6 = FullyConnected(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride= 2, bias= False, output_padding= 1)
        self.dense7 = FullyConnected(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride= 1)


    def forward(self, img_batch, t, label_batch= None):
        """
            Compute the forward pass of the U-Net Transformer.

            Steps:
                1. Embed the diffusion timestep t using Gaussian Fourier features.
                2. Embed the class label label_batch (if provided) and use it as context for 
                cross-attention in SpatialTransformer blocks.
                3. Pass the input through encoder convolutions, each conditioned with 
                a time-dependent Dense layer.
                4. Apply SpatialTransformer blocks at deeper layers to capture 
                global dependencies across spatial locations.
                5. Decode using transposed convolutions with skip connections.
                6. Normalize the predicted noise using the marginal σ(t).

            Args:
                img_batch (Tensor): 
                    Input noisy image x_t, shape (B, 1, H, W).
                t (Tensor): 
                    Diffusion timestep values, shape (B,).
                label_batch (Tensor, optional): 
                    Class labels, shape (B,). Used to compute class-conditioning 
                    via cross-attention. If None, self-attention only is used.

            Returns:
                Tensor:
                    Predicted noise ε_θ(img_batch_t, t, label_batch), shape (B, 1, H, W).
            
        """

        """----------------------------
            Time and label ENCODER
        ----------------------------"""
        t_embed = self.gelu(self.time_embed(t))
        label_embed = self.cond_embed(label_batch).unsqueeze(1)
        
        """----------------------------
                ENCODER BLOCK
        ----------------------------"""
        l1 = self.gelu(self.gnorm1(self.conv1(img_batch) + self.dense1(t_embed)))
        l2 = self.gelu(self.gnorm2(self.conv2(l1) + self.dense2(t_embed)))
        l3 = self.gelu(self.gnorm3(self.conv3(l2) + self.dense3(t_embed)))
        l3 = self.attn3(l3, label_embed)
        l4 = self.gelu(self.gnorm4(self.conv4(l3) + self.dense4(t_embed)))
        l4 = self.attn4(l4, label_embed)

        """----------------------------
                DECODER BLOCK
        ----------------------------"""
        d = self.gelu(self.tgnorm4(self.tconv4(l4) + self.dense5(t_embed)))
        d = self.gelu(self.tgnorm3(self.tconv3(d + l3) + self.dense6(t_embed)))
        d = self.gelu(self.tgnorm2(self.tconv2(d + l2) + self.dense7(t_embed)))
        d = self.tconv1(d + l1)

        # Normalize predicted noise by marginal std at time t. This is the score function in the paper.
        return d / self.marginal_prob_std(t)[:, None, None, None]

    @torch.no_grad()
    def sampling_technique(self, **kwargs):
        """
        Sample images from the diffusion model using Euler-Maruyama.

        kwargs può contenere:
            - num_steps
            - batch_size
            - x_shape
            - device
            - eps
            - y (class labels)
        """
        num_steps = kwargs.get("num_steps", 250)
        batch_size = kwargs.get("batch_size", 16)
        x_shape = kwargs.get("x_shape", (1, 28, 28))
        device = kwargs.get("device", "cuda")
        eps = kwargs.get("eps", 1e-3)
        y = kwargs.get("y", None)

        self.eval()
        with torch.no_grad():
            # Inizializza con rumore gaussiano
            t = torch.ones(batch_size, device= device)
            x = torch.randn(batch_size, *x_shape, device= device) * self.marginal_prob_std(t)[:, None, None, None]

            # Passi di tempo da 1 -> eps
            time_steps = torch.linspace(1.0, eps, num_steps, device= device)
            step_size = time_steps[0] - time_steps[1]

            for time_step in tqdm(time_steps, desc= "Sampling"):
                batch_time_step = torch.ones(batch_size, device= device) * time_step
                g = diffusion_coeff(batch_time_step)  # funzione globale
                eps_pred = self.forward(img_batch= x, t= batch_time_step, label_batch= y)
                mean_x = x + (g**2)[:, None, None, None] * eps_pred * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

            return mean_x.clamp(0.0, 1.0)