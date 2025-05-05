import os
import torch
import numpy as np
from tqdm.auto import tqdm, trange

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


class Trainer:
  def __init__(
    self,
    model,
    config,
    data_train,
    data_test,
    checkpoint_dir=None,
    resume_from=None,
    resume_from_run=None,
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
    self.start_step = 0
    self.num_steps = config.steps
    self.batch_size = config.batch_size
    self.checkpoint_freq = config.checkpoint_freq
    self.eval_freq = config.eval_freq
    self.log_freq = config.log_freq
    self.sample_freq = config.sample_freq
    self.sample_steps = config.sample_steps

    if resume_from is not None and checkpoint_dir:
      self.load_checkpoint(resume_from, resume_from_run)

  def load_checkpoint(self, step, resume_from_run):
    device = util.device()
    if resume_from_run:
      checkpoint_path = util.wandb_download_checkpoint(
        resume_from_run, step, self.checkpoint_dir
      )
    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step{step}.pt')
    if not os.path.exists(checkpoint_path):
      print(f'checkpoint for step {step} not found. retraining')
      return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    self.model.scorenet.load_state_dict(checkpoint['scorenet'])
    self.model.scorenet.to(device)
    self.opt.load_state_dict(checkpoint['optimizer'])
    self.scheduler.load_state_dict(checkpoint['scheduler'])
    self.start_step = checkpoint['step'] + 1
    print(f'resuming training from step {self.start_step}')

  def save_checkpoint(self, step):
    if not self.checkpoint_dir:
      return
    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step{step}.pt')
    directory = os.path.dirname(checkpoint_path)
    if directory:
      os.makedirs(directory, exist_ok=True)
    checkpoint = {
      'scorenet': self.model.scorenet.state_dict(),
      'optimizer': self.opt.state_dict(),
      'scheduler': self.scheduler.state_dict(),
      'step': step,
    }
    util.save(checkpoint, step, checkpoint_path)
    print(f'checkpoint at step {step} saved')

  def train(self):
    self.model.scorenet.train()
    device = util.device()
    data_iter = iter(self.data_train)
    for step in trange(self.start_step, self.num_steps):
      batch = next(data_iter, None)
      if batch is None:
        data_iter = iter(self.data_train)
        batch = next(data_iter)
      batch = batch.to(device)

      loss = loss_dwdse(self.model, batch)
      self.opt.zero_grad()
      loss.backward()
      if self.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
          self.model.scorenet.parameters(), max_norm=self.grad_clip
        )
      self.opt.step()
      self.scheduler.step()

      every_n = lambda ref: ref and step % ref == 0
      if every_n(self.checkpoint_freq) and step > 0:
        self.save_checkpoint(step)
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
      loss = loss_dwdse(self.model, batch)
      losses.append(loss.item())
    util.log({'test/loss': np.mean(losses), **log_extra})
