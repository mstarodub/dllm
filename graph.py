import torch


class AbsorbingGraph:
  def __init__(self, vocab_size):
    self.absorbing_state = vocab_size

  def sample_transition(self, x, sigma):
    move_chance = 1 - (-sigma).exp()
    # using independence (eq 13)
    move_idx = torch.rand(*x.shape, device=x.device) < move_chance
    x_perturbed = torch.where(move_idx, self.absorbing_state, x)
    return x_perturbed
