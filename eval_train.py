import os
import torch
from tqdm import tqdm

import util
from loss import loss_dwdse


def sample_t(batch_size):
  device = util.device()
  eps = 1e-3
  # almost t ~ U([0,1]), but remove dangerous 0 and 1 edges
  return (1 - eps) * torch.rand(batch_size, device=device) + eps


@torch.no_grad()
def eval(model, data, log_extra=dict()):
  model.scorenet.eval()
  device = util.device()
  for (batch,) in tqdm(data):
    batch = batch.to(device)
    t = sample_t(batch.shape[0])
    loss = loss_dwdse(model, batch, t)
    util.log({'test/loss': loss.item(), **log_extra})


class Trainer:
  def __init__(self, model, config, checkpoint_dir='checkpoints'):
    self.model = model
    self.opt = torch.optim.AdamW(model.scorenet.parameters(), lr=config.lr)
    self.checkpoint_dir = checkpoint_dir
    self.num_epochs = config.epochs
    self.batch_size = config.batch_size
    self.checkpoint_freq = config.snapshot_freq
    self.eval_freq = config.eval_freq
    self.log_freq = config.log_freq

  def train(self, data_train, data_test):
    self.model.scorenet.train()
    device = util.device()
    for epoch in range(self.num_epochs):
      for b, (batch,) in enumerate(data_train):
        batch = batch.to(device)
        t = sample_t(batch.shape[0])
        loss = loss_dwdse(self.model, batch, t)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        every_n = lambda ref: ref and b % ref == 0 and batch.shape[0] == self.batch_size
        if every_n(self.checkpoint_freq):
          torch.save(
            self.model.scorenet.state_dict(),
            os.path.join(self.checkpoint_dir, f'checkpoint_ep{epoch}_step{b}.pt'),
          )
        step_stats = {
          'epoch': epoch,
          'batch': epoch * len(data_train) + b,
        }
        if every_n(self.eval_freq):
          eval(self.model, data_test, step_stats)
        if every_n(self.log_freq):
          util.log({'train/loss': loss.item(), **step_stats})
