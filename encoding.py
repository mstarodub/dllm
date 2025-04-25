import torch
import torch.nn as nn
import math


# TODO: instead of this (use RoPE) and the sinusoidal timestep
# use the huggingface diffusers impl
# standard sinusoidal PE
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int):
    super().__init__()
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, seq_len: int) -> torch.Tensor:
    return self.pe[:, :seq_len]


class TimestepEncoding(nn.Module):
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    # Note: hidden_size corresponds to the main model's embed_dim
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
    )
    self.frequency_embedding_size = frequency_embedding_size

  # sinusoidal timestep embedding, based on OpenAI GLIDE implementation:
  # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
  @staticmethod
  def timestep_embedding(t, out_dim):
    max_period = 10_000
    half = out_dim // 2
    freqs = torch.exp(
      -math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]  # Shape: [batch_size, half]
    embedding = torch.cat(
      [torch.cos(args), torch.sin(args)], dim=-1
    )  # Shape: [batch_size, out_dim]
    if out_dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding  # Shape: [batch_size, out_dim]

  def forward(self, t):
    # t is expected to be shape [batch_size] or [batch_size, 1]
    t_freq = self.timestep_embedding(t.squeeze(), self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)  # Shape: [batch_size, hidden_size]
    return t_emb
