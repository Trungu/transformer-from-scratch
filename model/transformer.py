import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layers import LayerNorm, FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_embed):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_embed)
        self.ffn = FeedForward(d_embed)
        
        self.norm1 = LayerNorm(d_embed)
        self.norm2 = LayerNorm(d_embed)

    def forward(self, X):
        x1 = X + self.attention(self.norm1(X))
        x2 = x1 + self.ffn(self.norm2(x1))

        return x2
