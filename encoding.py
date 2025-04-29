import torch
import torch.nn as nn
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int):
    super().__init__()
    positions = torch.arange(max_len, dtype=torch.float32)
    pe = (
      get_1d_sincos_pos_embed_from_grid(
        embed_dim=d_model, pos=positions, output_type='pt'
      )
      .unsqueeze(0)
      .float()
    )
    self.register_buffer('pe', pe)

  def forward(self, seq_len: int):
    return self.pe[:, :seq_len]


class TimestepEncoding(nn.Module):
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
    )
    self.frequency_embedding_size = frequency_embedding_size

  def forward(self, t):
    # t: [B] or [B,1]
    if t.dim() > 1 and t.size(-1) == 1:
      t = t.squeeze(-1)
    emb = get_timestep_embedding(
      timesteps=t,
      embedding_dim=self.frequency_embedding_size,
      scale=1,
    )
    return self.mlp(emb)
