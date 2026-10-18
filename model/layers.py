import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_embed, 256),
            nn.ReLU(),
            nn.Linear(256, d_embed)
        )

    def forward(self, X):
        out = self.fc(X)
        return out


class LayerNorm(nn.Module):
    def __init__(self, d_embed, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_embed))
        self.beta = nn.Parameter(torch.zeros(d_embed))
        self.eps = eps

    def forward(self,x):
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True, correction=0)

        return ((x - mean) / torch.sqrt(variance + self.eps)) * self.gamma + self.beta

