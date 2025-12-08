import torch
import torch.nn as nn
from tqdm import tqdm
from .utils.gaussian_projection import GaussianFourierProjection
from .utils.fully_connected import FullyConnected
from .spatial_transformer import SpatialTransformer
from .utils.probability import diffusion_coeff, marginal_prob_std

class DiffusionModel(nn.Module):
    """
    U-Net adattata per supportare input multi-risoluzione e multi-canale 
    (es. MNIST 1x28x28 e CIFAR-10 3x32x32).
    """
    def __init__(self, 
                 marginal_prob_std=marginal_prob_std, 
                 channels=[32, 64, 128, 256], 
                 embed_dim=256,
                 text_dim=256, 
                 class_num=10,
                 in_channels=1,  
                 img_size=28):      
        
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Continuous time embedding
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Class embedding
        self.cond_embed = nn.Embedding(class_num, text_dim)

        self.gelu = nn.GELU()
        self.marginal_prob_std = marginal_prob_std
        
        # Calcolo dinamico dell'output padding per il decoder.
        # CIFAR (32px): Deepest layer 4->8 (op=1)
        # MNIST (28px): Deepest layer 4->7 (op=0)
        deep_output_padding = 1 if img_size == 32 else 0

        """------------------------
            ENCODING BLOCK
        ------------------------"""
        # Aggiunto padding=1 per mantenere le dimensioni spaziali controllate
        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = FullyConnected(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = FullyConnected(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = FullyConnected(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = FullyConnected(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)

        """---------------------
            DECODING BLOCK
        ----------------------"""
        # tconv4 usa deep_output_padding variabile
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, padding=1, output_padding=deep_output_padding)
        self.dense5 = FullyConnected(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        # tconv3 e tconv2 usano output_padding=1 standard per raddoppiare (es. 8->16 o 7->14)
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, padding=1, output_padding=1)
        self.dense6 = FullyConnected(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, padding=1, output_padding=1)
        self.dense7 = FullyConnected(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        # L'ultimo layer deve ritornare al numero di canali di input (es. 3 per CIFAR)
        self.tconv1 = nn.ConvTranspose2d(channels[0], in_channels, 3, stride=1, padding=1)

    def forward(self, img_batch, t, label_batch=None):
        # Il forward rimane identico strutturalmente
        t_embed = self.gelu(self.time_embed(t))
        label_embed = self.cond_embed(label_batch).unsqueeze(1)
        
        l1 = self.gelu(self.gnorm1(self.conv1(img_batch) + self.dense1(t_embed)))
        l2 = self.gelu(self.gnorm2(self.conv2(l1) + self.dense2(t_embed)))
        l3 = self.gelu(self.gnorm3(self.conv3(l2) + self.dense3(t_embed)))
        l3 = self.attn3(l3, label_embed)
        l4 = self.gelu(self.gnorm4(self.conv4(l3) + self.dense4(t_embed)))
        l4 = self.attn4(l4, label_embed)

        d = self.gelu(self.tgnorm4(self.tconv4(l4) + self.dense5(t_embed)))
        d = self.gelu(self.tgnorm3(self.tconv3(d + l3) + self.dense6(t_embed)))
        d = self.gelu(self.tgnorm2(self.tconv2(d + l2) + self.dense7(t_embed)))
        d = self.tconv1(d + l1)

        return d / self.marginal_prob_std(t)[:, None, None, None]

    @torch.no_grad()
    def sampling_technique(self, **kwargs):
        # Aggiornati i default per usare i parametri dell'istanza se non specificati
        num_steps = kwargs.get("num_steps", 250)
        batch_size = kwargs.get("batch_size", 16)
        
        # Default shape dinamica basata su init
        default_shape = (self.in_channels, self.img_size, self.img_size)
        x_shape = kwargs.get("x_shape", default_shape)
        
        device = kwargs.get("device", "cuda")
        eps = kwargs.get("eps", 1e-3)
        y = kwargs.get("y", None)

        self.eval()
        with torch.no_grad():
            t = torch.ones(batch_size, device=device)
            x = torch.randn(batch_size, *x_shape, device=device) * self.marginal_prob_std(t)[:, None, None, None]
            
            time_steps = torch.linspace(1.0, eps, num_steps, device=device)
            step_size = time_steps[0] - time_steps[1]

            for time_step in tqdm(time_steps, desc="Sampling"):
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                g = diffusion_coeff(batch_time_step)
                eps_pred = self.forward(img_batch=x, t=batch_time_step, label_batch=y)
                mean_x = x + (g**2)[:, None, None, None] * eps_pred * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

            return mean_x.clamp(0.0, 1.0)