import os
import torch
import numpy as np
from tqdm import tqdm

import util
import score
import noise
import graph
from loss import loss_dwdse
from reverse import sample_log


class Sedd:
  def __init__(self, config, tokenizer):
    device = util.device()
    self.scorenet = score.ScoreNet(
      vocab_size=tokenizer.vocab_size,
      embed_dim=config.embed_dim,
      time_embed_dim=config.time_embed_dim,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      max_seq_len=config.max_seq_len,
      pad_idx=tokenizer.pad_idx,
      dropout=config.dropout_p,
    ).to(device)
    self.graph = graph.AbsorbingGraph(tokenizer.vocab_size)
    self.noise = noise.LogLinearNoise()
    self.tokenizer = tokenizer

  def save(self, path):
    directory = os.path.dirname(path)
    if directory:
      os.makedirs(directory, exist_ok=True)
    torch.save(self.scorenet.state_dict(), path)

  def load(self, path):
    device = util.device()
    if os.path.exists(path):
      self.scorenet.load_state_dict(torch.load(path, map_location=device))
      self.scorenet.to(device)


class Trainer:
  def __init__(
    self,
    model,
    config,
    data_train,
    data_test,
    checkpoint_dir=None,
  ):
    self.model = model
    self.data_train = data_train
    self.data_test = data_test
    self.opt = torch.optim.AdamW(model.scorenet.parameters(), lr=config.lr)
    self.checkpoint_dir = checkpoint_dir
    self.num_epochs = config.epochs
    self.batch_size = config.batch_size
    self.checkpoint_freq = config.snapshot_freq
    self.eval_freq = config.eval_freq
    self.log_freq = config.log_freq
    self.sample_freq = config.sample_freq
    self.sample_steps = config.sample_steps

  def train(self):
    self.model.scorenet.train()
    device = util.device()
    for epoch in range(self.num_epochs):
      for b, batch in enumerate(self.data_train):
        batch = batch.to(device)
        t = sample_t(batch.shape[0], self.model.noise.eps)
        loss = loss_dwdse(self.model, batch, t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        every_n = lambda ref: ref and b % ref == 0 and batch.shape[0] == self.batch_size
        if every_n(self.checkpoint_freq) and b > 0 and self.checkpoint_dir:
          self.model.save(
            os.path.join(self.checkpoint_dir, f'checkpoint_ep{epoch}_step{b}.pt')
          )
        step_stats = {
          'epoch': epoch,
          'batch': epoch * len(self.data_train) + b,
        }
        if every_n(self.eval_freq):
          self.eval(step_stats)
          self.model.scorenet.train()
        if every_n(self.log_freq):
          util.log({'train/loss': loss.item(), **step_stats})
        if every_n(self.sample_freq):
          sample_log(self.model, self.sample_steps, step_stats)

  @torch.no_grad()
  def eval(self, log_extra=dict()):
    self.model.scorenet.eval()
    device = util.device()
    losses = []
    for batch in tqdm(self.data_test):
      batch = batch.to(device)
      t = sample_t(batch.shape[0], self.model.noise.eps)
      loss = loss_dwdse(self.model, batch, t)
      losses.append(loss.item())
    util.log({'test/loss': np.mean(losses), **log_extra})


def sample_t(batch_size, eps):
  device = util.device()
  # almost t ~ U([0,1]), but remove dangerous 0 and 1 edges
  return (1 - eps) * torch.rand(batch_size, device=device) + eps
