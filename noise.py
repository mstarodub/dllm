import torch
import torch.nn as nn


class LogLinearNoise(nn.Module):
  # TODO: explain this, cleanup
  """
  Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
  when t goes from 0 to 1. Used for absorbing

  Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
  """

  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  # Rate of change of noise ie g(t)
  def noise_rate(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  # Total noise ie \int_0^t g(t) dt (+ g(0) ?)
  def noise_total(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def forward(self, t):
    return (self.noise_total(t).unsqueeze(1), self.noise_rate(t).unsqueeze(1))

  @staticmethod
  def t_eps():
    return 1e-3
