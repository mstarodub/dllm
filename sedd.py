import os
import torch
import numpy as np
from tqdm import tqdm, trange

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
    self.scorenet = torch.compile(self.scorenet)
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
    self.lr = config.lr
    self.grad_clip = config.grad_clip
    self.warmup = config.warmup
    self.opt = torch.optim.AdamW(model.scorenet.parameters(), lr=self.lr)
    self.scheduler = torch.optim.lr_scheduler.LambdaLR(
      self.opt,
      lr_lambda=lambda step: min(step / self.warmup, 1.0) if self.warmup else 1.0,
    )
    self.checkpoint_dir = checkpoint_dir
    self.num_steps = config.steps
    self.batch_size = config.batch_size
    self.checkpoint_freq = config.snapshot_freq
    self.eval_freq = config.eval_freq
    self.log_freq = config.log_freq
    self.sample_freq = config.sample_freq
    self.sample_steps = config.sample_steps

  def train(self):
    self.model.scorenet.train()
    device = util.device()
    data_iter = iter(self.data_train)
    for step in trange(self.num_steps):
      batch = next(data_iter, None)
      if batch is None:
        data_iter = iter(self.data_train)
        batch = next(data_iter)
      batch = batch.to(device)

      t = sample_t(batch.shape[0], self.model.noise.eps)
      loss = loss_dwdse(self.model, batch, t)
      self.opt.zero_grad()
      loss.backward()
      if self.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
          self.model.scorenet.parameters(), max_norm=self.grad_clip
        )
      self.opt.step()
      self.scheduler.step()

      every_n = lambda ref: ref and step % ref == 0
      if every_n(self.checkpoint_freq) and step > 0 and self.checkpoint_dir:
        self.model.save(os.path.join(self.checkpoint_dir, f'checkpoint_step{step}.pt'))
      step_stats = {'step': step}
      if every_n(self.eval_freq):
        self.eval(step_stats)
        self.model.scorenet.train()
      if every_n(self.log_freq) and batch.shape[0] == self.batch_size:
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
