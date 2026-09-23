from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_history=50, hidden_dim=128, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder, self.norm = nn.TransformerEncoder(encoder_layer, layers), nn.LayerNorm(hidden_dim)

    def encode(self, histories: Tensor) -> Tensor:
        length = histories.size(1)
        positions = torch.arange(length, device=histories.device).unsqueeze(0)
        hidden = self.item_embedding(histories) * math.sqrt(self.item_embedding.embedding_dim) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=histories.device), diagonal=1)
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=histories.eq(0))
        last = histories.ne(0).sum(1).clamp_min(1) - 1
        return self.norm(hidden[torch.arange(histories.size(0), device=histories.device), last])

    def forward(self, histories: Tensor) -> Tensor:
        return self.encode(histories) @ self.item_embedding.weight.T

    @torch.no_grad()
    def topk(self, histories: Tensor, k=20) -> tuple[Tensor, Tensor, Tensor]:
        logits = self(histories); logits[:, 0] = -torch.inf
        scores, items = logits.topk(k, dim=-1)
        probs = scores.softmax(-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        return items, scores, entropy
