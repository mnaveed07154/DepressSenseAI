import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x, mask=None):
        scores = self.scorer(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights
