import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    """
        This class transforms a scalar to an high dimensional embedding using sinusoidal function.
        It is used to make the model understand the temporal step t in a feature vector.
        It is the same idea of positional encoding in `Attention is all you need` paper.
    """
    def __init__(self, embed_dim, scale= 30.):
        """
            Args:
                embed_dim: final dimension of the embedding wanted
                scale (optional): random scale to initialize projection weights
        """
        super().__init__()
        
        # W is a vector of random frequencies of embed_dim//2 dim * scale, 
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, t):
        """Transforms a scalar diffusion time t ∈ [0,1] into a high dimenional embedding using
            sinusoidal functions to make patterns. 

            Args:
                t (float) : scalar diffusion time of shape (B,)

            Returns:
                torch.tensor : high dimensional embedding time of shape (B, embed_dim) 
        """

        # add a dimension to the time -> (B,1) and to W (1, embedd_dim//2) ---> (B, D)
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi #[batch_size, embedd_dim//2] scaling all frequencies in [0, 2pi]
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1) # [B, embed_dim]