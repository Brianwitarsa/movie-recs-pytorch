# recs/models.py
import torch
import torch.nn as nn

class MF(nn.Module):
    """Simple Matrix Factorization with user/item biases."""
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.ubias = nn.Embedding(n_users, 1)
        self.ibias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        nn.init.zeros_(self.ubias.weight)
        nn.init.zeros_(self.ibias.weight)

    def forward(self, u, i):
        pu = self.user(u)            # [B, dim]
        qi = self.item(i)            # [B, dim]
        dot = (pu * qi).sum(dim=1)   # [B]
        return dot + self.ubias(u).squeeze(1) + self.ibias(i).squeeze(1)
