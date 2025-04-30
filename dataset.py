import string
from itertools import chain
from Bio import SeqIO
import torch
from torch.utils.data import DataLoader

import util


def get_acyp_sequences():
  with open('data/hypf.fa', 'r') as f:
    return [str(rec.seq) for rec in SeqIO.parse(f, 'fasta')]


def ascii_alphabet():
  return list(string.ascii_lowercase) + [' ', '.', ',', '!', '?']


def protein_alphabet():
  # https://en.wikipedia.org/wiki/Proteinogenic_amino_acid
  # X for unknown amino acid
  return list('ACDEFGHIKLMNPQRSTVWXY')


def mk_collate(pad_idx, max_seq_len):
  device = util.device()

  def collate(batch):
    if max_seq_len is None:
      return [torch.tensor(seq, device=device) for seq in batch]
    out = []
    for seq in batch:
      if len(seq) > max_seq_len:
        seq = seq[:max_seq_len]
      else:
        seq = seq + [pad_idx] * (max_seq_len - len(seq))
      out.append(seq)
    return torch.tensor(out, device=device)

  return collate


def chunk(tokenized, block_size):
  flat = list(chain.from_iterable(tokenized))
  return [
    flat[i * block_size : (i + 1) * block_size] for i in range(len(flat) // block_size)
  ]


def acyp_dataset(
  tokenizer,
  batch_size,
  max_seq_len,
  block_size=None,
):
  device = util.device()
  seqs = get_acyp_sequences()
  tokenized = tokenizer.encode_all(seqs)
  if block_size is not None:
    tokenized = chunk(tokenized, block_size)
    collate_fn = lambda batch: torch.tensor(batch, device=device)
  else:
    collate_fn = mk_collate(tokenizer.pad_idx, max_seq_len)
  return DataLoader(
    tokenized,
    batch_size=batch_size,
    shuffle=True,
    num_workers=util.dl_workers(),
    collate_fn=collate_fn,
  )


def text_dataset(
  tokenizer,
  strings,
  batch_size,
  max_seq_len,
  block_size=None,
):
  encoded = tokenizer.encode_all(strings)
  collate_fn = mk_collate(tokenizer.pad_idx, max_seq_len)
  return DataLoader(
    encoded,
    batch_size=batch_size,
    shuffle=True,
    num_workers=util.dl_workers(),
    collate_fn=collate_fn,
  )
