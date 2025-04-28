import torch
import torch.nn as nn
import torch.nn.functional as F
import string
from typing import List

import encoding


class CharTokenizer:
  def __init__(self):
    self.chars = list(string.ascii_lowercase) + [' ', '.', ',', '!', '?']

    # this is not nice from a theory standpoint. we could fill every input
    # with spaces instead of PADs, and that way it would be recognized by
    # the markov process as changeable states. via choosing a good dataset
    # we can still have variable answer sizes (the model may learn to insert spaces)
    self.pad_token = '[PAD]'
    self.vocab = [self.pad_token] + self.chars
    self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
    self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
    self.pad_idx = self.char_to_idx[self.pad_token]

  @property
  def vocab_size(self) -> int:
    return len(self.vocab)

  def encode(self, text: str) -> List[int]:
    def enc_tf(ch):
      idx = self.char_to_idx.get(ch)
      if idx is None:
        raise ValueError(f'char {ch} is not in vocabulary')
      return idx

    return [enc_tf(char) for char in text]

  def decode(self, indices: List[int], skip_special_tokens=True) -> str:
    chars = []
    for idx in indices:
      char = self.idx_to_char.get(idx, '□')
      if char == self.pad_token:
        # ignore PAD if skipping
        if skip_special_tokens:
          continue
        # stop decoding
        else:
          break
      chars.append(char)
    return ''.join(chars)


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
    return F.softplus(logits)
