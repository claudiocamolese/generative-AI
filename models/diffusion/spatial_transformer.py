import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .transformer import Transformer

class SpatialTransformer(nn.Module):
    """
        This class is a Transformer block applied to the 2D images features map.
        It includes residual connection.
    """
    def __init__(self, hidden_dim, context_dim):
        """
            Initializes a Spatial Transformer block that applies a Transformer to 2D feature maps.

            Args:
                hidden_dim (int): Number of channels in the input feature map (token dimension).
                context_dim (int): Dimension of context tokens for cross-attention. If None,
                                the block performs self-attention only.

            Components:
                - self.transformer: Transformer block that handles self- and cross-attention
                                    and a feed-forward network (FFN) on flattened spatial tokens.
        """
        super().__init__()
        self.transformer = Transformer(hidden_dim, context_dim)

    def forward(self, x, context= None):
        """
            Forward pass of the Spatial Transformer.

            Args:
                x (torch.Tensor): Input feature map of shape (batch_size, channels, height, width).
                context (torch.Tensor, optional): External context tensor for cross-attention,
                                                of shape (batch_size, seq_len_context, context_dim).
                                                If None, only self-attention is applied.

            Returns:
                torch.Tensor: Output feature map of the same shape as input (batch_size, channels, height, width),
                            with features updated via attention across spatial positions.
        """
        b, c, h, w = x.shape
        x_in = x
        # change the sequence of tokens from (batch, channels, height, width) to (batch, seq_len=h*w, channels)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        # returns to the original sequence of the feature map 2D original
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in # (batch, channels, height, width) spatial features enriched by self/cross attention