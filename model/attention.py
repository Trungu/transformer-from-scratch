import torch
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, count_heads, d_embed):
        super().__init__()
        self.heads = count_heads
        self.d_embed = d_embed
        self.d_head = d_embed // count_heads

        self.w_q = nn.ParameterList(
            [nn.Parameter(torch.randn(d_embed, self.d_head)) for _ in range(self.heads)])

        self.w_k = nn.ParameterList(
                [nn.Parameter(torch.randn(d_embed, self.d_head)) for _ in range(self.heads)])

        self.w_v = nn.ParameterList(
                [nn.Parameter(torch.randn(d_embed, self.d_head)) for _ in range(self.heads)])
        
        self.wO = nn.Parameter(torch.randn(d_embed, d_embed))


    def forward(self, X):
        query_vector: list = [torch.matmul(X, w_q) for w_q in self.w_q]
        key_vector: list = [torch.matmul(X, w_k) for w_k in self.w_k]
        value_vector: list = [torch.matmul(X, w_v) for w_v in self.w_v]
        T: int = X.size(0)

        head_outputs = [self.__attention(query_vector[i], key_vector[i], value_vector[i], T) for i in range(self.heads)]
        attention_concat = torch.cat(head_outputs, dim=1)

        return torch.matmul(attention_concat, self.wO)


    def __attention(self, q_i, k_i, v_i, T):
        scaled_dot_product = torch.matmul(q_i, k_i.T) / math.sqrt(self.d_head)

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool),
            diagonal=1
        )
        masked_scores = scaled_dot_product.masked_fill(mask, float("-inf"))

        soft_max = nn.Softmax(dim=1)
        out = soft_max(masked_scores)

        attention_matrix = torch.matmul(out, v_i)

        return attention_matrix

