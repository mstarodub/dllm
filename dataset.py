import string
from itertools import chain
from Bio import SeqIO
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, IterableDataset

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
      return torch.tensor(batch, device=device)
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
  block_size,
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
    # num_workers=util.dl_workers(),
    collate_fn=collate_fn,
  )


def text_dataset(
  tokenizer,
  strings,
  batch_size,
  max_seq_len,
  block_size,
):
  encoded = tokenizer.encode_all(strings)
  collate_fn = mk_collate(tokenizer.pad_idx, max_seq_len)
  return DataLoader(
    encoded,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=util.dl_workers(),
    collate_fn=collate_fn,
  )


def gpt2_dataset(
  tokenizer,
  batch_size,
  block_size,
):
  device = util.device()

  dataset_dir = 'data/tinystories'
  train_ds = load_dataset(
    'roneneldan/TinyStories',
    split='train',
    streaming=True,
    cache_dir=dataset_dir,
  )
  test_ds = load_dataset(
    'roneneldan/TinyStories',
    split='validation',
    streaming=True,
    cache_dir=dataset_dir,
  )

  class StreamingDataset(IterableDataset):
    def __init__(self, raw_dataset, tokenizer, block_size):
      self.raw_dataset = raw_dataset
      self.tokenizer = tokenizer
      self.block_size = block_size

    def __iter__(self):
      buffer = []
      for example in self.raw_dataset:
        tokens = self.tokenizer.encode_all([example['text']])[0]
        buffer.extend(tokens)
        while len(buffer) >= self.block_size:
          chunk = buffer[: self.block_size]
          buffer = buffer[self.block_size :]
          yield torch.tensor(chunk, device=device)

  train_loader = DataLoader(
    StreamingDataset(train_ds, tokenizer, block_size),
    batch_size=batch_size,
  )

  test_loader = DataLoader(
    StreamingDataset(test_ds, tokenizer, block_size),
    batch_size=batch_size,
  )

  return train_loader, test_loader
