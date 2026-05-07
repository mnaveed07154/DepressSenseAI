import torch
import torch.nn as nn
from .fusion import GatedFeatureFusion
from .attention_pooling import AttentionPooling
from .gradient_reversal import grad_reverse

class VoxDepressNet(nn.Module):
    def __init__(self, ssl_dim=768, acoustic_dim=64, fusion_dim=256, temporal_hidden=128,
                 temporal_layers=1, attention_dim=128, num_classes=2, num_domains=3, dropout=0.3,
                 encoder_type="bigru"):
        super().__init__()
        self.fusion = GatedFeatureFusion(ssl_dim, acoustic_dim, fusion_dim, dropout)
        self.encoder_type = encoder_type
        if encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4, dim_feedforward=fusion_dim*2,
                                               dropout=dropout, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=max(1, temporal_layers))
            enc_dim = fusion_dim
        else:
            self.temporal = nn.GRU(fusion_dim, temporal_hidden, num_layers=temporal_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if temporal_layers > 1 else 0.0)
            enc_dim = temporal_hidden * 2
        self.attention = AttentionPooling(enc_dim, attention_dim)
        self.classifier = nn.Sequential(nn.Linear(enc_dim, enc_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(enc_dim//2, num_classes))
        self.severity = nn.Sequential(nn.Linear(enc_dim, enc_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(enc_dim//2, 1))
        self.domain = nn.Sequential(nn.Linear(enc_dim, enc_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(enc_dim//2, num_domains))

    def forward(self, ssl, acoustic, mask=None, grl_lambda=1.0):
        fused, gate = self.fusion(ssl, acoustic)
        if self.encoder_type == "transformer":
            key_padding_mask = None if mask is None else ~mask
            encoded = self.temporal(fused, src_key_padding_mask=key_padding_mask)
        else:
            encoded, _ = self.temporal(fused)
        session, attn = self.attention(encoded, mask)
        class_logits = self.classifier(session)
        sev = self.severity(session).squeeze(-1)
        domain_logits = self.domain(grad_reverse(session, grl_lambda))
        return {"class_logits": class_logits, "severity": sev, "domain_logits": domain_logits,
                "embedding": session, "attention": attn, "gate": gate}
