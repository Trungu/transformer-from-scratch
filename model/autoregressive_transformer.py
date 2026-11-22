import math
import torch
import torch.nn as nn
from .transformer import TransformerBlock, LayerNorm

class AutoRegressiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_embed, num_heads, max_context_length, num_blocks=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.d_head = d_embed // num_heads

        self.w_vocab = nn.Parameter(
            torch.randn(d_embed, vocab_size) / math.sqrt(d_embed))

        self.bias = nn.Parameter(
                torch.zeros(vocab_size))

        self.token_embeddings = nn.Parameter(
            torch.randn(vocab_size, d_embed) / math.sqrt(d_embed))
        
        self.pos_embeddings = nn.Parameter(
            torch.randn(max_context_length, d_embed) / math.sqrt(d_embed))

        self.layers = nn.ModuleList(
            [TransformerBlock(self.num_heads, self.d_embed) for _ in range(num_blocks)])

        self.transformer_sequence = nn.Sequential(*self.layers)
        self.layer_norm_final = LayerNorm(d_embed)


    def forward(self, tokens):
        embedded_tokens = self.token_embeddings[tokens] + self.pos_embeddings[:tokens.size(-1)]

        output = self.transformer_sequence(embedded_tokens)
        output = self.layer_norm_final(output)

        logits = torch.matmul(output, self.w_vocab) + self.bias

        return logits;
