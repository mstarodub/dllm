import torch
import torch.nn as nn

import encoding


class ScoreNet(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    embed_dim: int,
    time_embed_dim: int,
    num_heads: int,
    num_layers: int,
    max_seq_len: int,
    pad_idx: int,
    dropout: float,
  ):
    super().__init__()
    self.embed_dim = embed_dim
    self.pad_idx = pad_idx
    self.max_seq_len = max_seq_len
    # +1 for absorbing state
    vocab_size = vocab_size + 1

    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_idx)
    self.pos_encoder = encoding.PositionalEncoding(embed_dim, max_len=max_seq_len)
    self.time_encoder = encoding.TimestepEncoding(embed_dim, time_embed_dim)
    self.dropout_layer = nn.Dropout(p=dropout)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim,
      nhead=num_heads,
      dim_feedforward=4 * embed_dim,
      dropout=dropout,
      activation='gelu',
      batch_first=True,
      norm_first=True,
    )
    self.transformer = nn.TransformerEncoder(
      encoder_layer=encoder_layer, num_layers=num_layers
    )
    self.output_layer = nn.Linear(embed_dim, vocab_size)

  def forward(self, src: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = src.shape
    src_padding_mask = src == self.pad_idx
    token_embedded = self.embedding(src)
    pos_encoding = self.pos_encoder(seq_len)
    time_encoding = self.time_encoder(sigma).unsqueeze(1)
    x = self.dropout_layer(token_embedded + pos_encoding + time_encoding)
    transformer_out = self.transformer(src=x, src_key_padding_mask=src_padding_mask)
    logits = self.output_layer(transformer_out)
    return logits
