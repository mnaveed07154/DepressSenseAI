import torch
import torch.nn as nn

class GatedFeatureFusion(nn.Module):
    def __init__(self, ssl_dim=768, acoustic_dim=64, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.ssl_proj = nn.Sequential(nn.Linear(ssl_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.acoustic_proj = nn.Sequential(nn.Linear(acoustic_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())

    def forward(self, ssl_x, acoustic_x):
        s = self.ssl_proj(ssl_x)
        a = self.acoustic_proj(acoustic_x)
        g = self.gate(torch.cat([s, a], dim=-1))
        fused = g * s + (1.0 - g) * a
        return fused, g
