import torch
import torch.nn as nn


# TODO: explain this
class LogLinearNoise(nn.Module):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def noise_rate(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  # \int_0^T \sigma(t) dt
  def noise_total(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def forward(self, t):
    return (
      self.noise_total(t).unsqueeze(1),
      self.noise_rate(t).unsqueeze(1),
    )
