import os
import torch
import wandb


def device():
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dl_workers():
  return 12 if torch.cuda.is_available() else 0


def grad_debug():
  torch.autograd.set_detect_anomaly(True)


def settings():
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def log(log_dict):
  if wandb.run is not None:
    wandb.log(log_dict)
  else:
    print(*log_dict.items(), sep='\n')


class Config:
  def __init__(self, d):
    self.cf_dict = d
    if d is not None:
      for key, value in d.items():
        setattr(self, key, value)
