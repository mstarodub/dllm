import os
import torch
from tqdm import tqdm

import util


@torch.no_grad()
def eval(model, data):
  model.eval()
  device = util.device()
  for batch in tqdm(data):
    batch = batch.to(device)
    util.log(...)


class Trainer:
  def __init__(self, model, config, checkpoint_dir='checkpoints'):
    self.model = model
    self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
    self.checkpoint_dir = checkpoint_dir
    self.num_epochs = config.epochs
    self.batch_size = config.batch_size
    self.checkpoint_freq = config.snapshot_freq
    self.eval_freq = config.eval_freq
    self.log_freq = config.log_freq

  def train(self, data_train, data_test):
    self.model.train()
    device = util.device()
    for epoch in self.num_epochs:
      for b, batch in enumerate(data_train):
        batch = batch.to(device)
        loss = ...
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        every_n = lambda ref: b % ref == 0 and batch.shape[0] == self.batch_size
        if every_n(self.checkpoint_freq):
          torch.save(
            self.model.state_dict(), os.path.join(self.checkpoint_dir, 'checkpoint.pt')
          )
        if every_n(self.eval_freq):
          eval(self.model, data_test)
        if every_n(self.log_freq):
          util.log(...)
