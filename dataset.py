import string
from Bio import SeqIO
from torch.utils.data import DataLoader, TensorDataset


def acyp_dataset(batch_size):
  with open('data/hypf.fa', 'r') as f:
    sequences = [str(rec.seq) for rec in SeqIO.parse(f, 'fasta')]
  return DataLoader(TensorDataset(sequences), batch_size=batch_size, shuffle=True)


def ascii_alphabet():
  return list(string.ascii_lowercase) + [' ', '.', ',', '!', '?']


def protein_alphabet():
  # https://en.wikipedia.org/wiki/Proteinogenic_amino_acid
  return list('ACDEFGHIKLMNPQRSTVWY')
