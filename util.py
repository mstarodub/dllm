import torch
import wandb
# TODO: use wandb config / log with wandb if enabled


def device():
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(log_dict):
  print(*log_dict.items(), sep='\n')


class Config:
  def __init__(self, d):
    if d is not None:
      for key, value in d.items():
        setattr(self, key, value)
