import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
        This class implements a cross-attention mechanism which reduced to self-attention when context is not passed
        As in attention, each token produce a query, key, value and the attention is computed.
        Compute `softmax` to get scores.

        - cross_attention:
            each token attends all the other tokens in the same sequence to update its value
            (Q, K, V) comes from the same input
        
        - self_attention:
            a set of tokens attend another set of tokens to update its value
            Q (batch_size, seq_len, hidden_dim) and (K / V) (batch_size, seq_len_context, context_dim)
    """

    def __init__(self, embed_dim, hidden_dim, context_dim= None):
        """
        Args:
            embed_dim: dimension of the input token
            hidden_dim: dimension of the Q,K in the attention space
            context_dim (optional): context dimension. If it is None -> self-attention
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.query = nn.Linear(hidden_dim, embed_dim, bias= False)

        if context_dim is None:
            """---------------------- 
                SELF-ATTENTION
            ----------------------"""
            self.self_attention = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias= False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias= False)

        else:
            """---------------------- 
                CROSS-ATTENTION
            ----------------------"""
            self.self_attention = False
            self.key = nn.Linear(context_dim, embed_dim, bias= False)
            self.value = nn.Linear(context_dim, hidden_dim, bias= False)

    def forward(self, tokens, context= None):
        """Outputs Q, K, V

        Args:
            tokens: input tokens (batch_size, seq_len, hidden_dim)
            context (optional): context tokens (batch_size, seq_len_context, context_dim). Defaults to None.

        Returns:
            (batch, seq_len_query, hidden_dim): attention scores
        """
        if self.self_attention:
            Q, K, V = self.query(tokens), self.key(tokens), self.value(tokens)
        else:
            Q, K, V = self.query(tokens), self.key(context), self.value(context)
        
        attention_scores = torch.einsum('bth,bsh->bts', Q, K)
        attention_mats = F.softmax(attention_scores, dim= -1) 
        context_vector = torch.einsum("bts,bsh->bth", attention_mats, V)
        return context_vector