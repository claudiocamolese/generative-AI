import torch.nn as nn
from .cross_attention import CrossAttention

class Transformer(nn.Module):
    """
        This class implements a Transformer block using CrossAttention if context exists, otherwise SelfAttention     
    """
    def __init__(self, hidden_dim, context_dim):
        """
        Args:
            hidden_dim (int): Dimension of input and output tokens.
            context_dim (int): Dimension of context tokens for cross-attention.
                               If None, cross-attention can behave like self-attention.

        Components:
            - self.attn_self: self-attention block applied to the tokens themselves.
            - self.attn_cross: cross-attention block between tokens and external context.
            - self.norm1, norm2, norm3: LayerNorms applied before each sub-layer.
            - self.ffn: feed-forward network (MLP with 4x expansion and GELU activation).
        """
        super(Transformer, self).__init__()
        # self-attention of the tokens itselfs
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)
        # cross-attention with context esterno
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x, context=None):
        """
            Forward pass of the Transformer block.

            Args:
                x : Input token tensor of shape (batch_size, seq_len, hidden_dim).
                context (optional): External context tensor for cross-attention,
                                                of shape (batch_size, seq_len_context, context_dim).
                                                If None, cross-attention behaves like self-attention.

            Returns:
                torch.Tensor: Updated token tensor of shape (batch_size, seq_len, hidden_dim).
                            Each token contains combined information from self-attention,
                            cross-attention, and the feed-forward network.
        """
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        return x

