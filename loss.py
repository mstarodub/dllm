import torch

# TODO: broken. using the reference loss for now
# integral via single MC sample of t
# def loss_dwdse(model, x0, t):
#   sigma, dsigma = model.noise(t)

#   esigm1 = torch.where(
#     sigma < 0.5,
#     torch.expm1(sigma),
#     torch.exp(sigma) - 1,
#   )
#   # expand_as to make the sigma scalar equal for every token
#   ratio = (1.0 / esigm1).expand_as(x0)

#   xt = model.graph.sample_transition(x0, sigma)
#   nonpadded_mask = x0 != model.scorenet.pad_idx
#   nonabsorbed_mask = xt != model.graph.absorbing_state
#   scores = model.scorenet(xt, sigma)
#   # (Batch, seqLen, Vocabsize) -> (B, L):
#   # using the MASK column we get the last row - all other contributions
#   # are 0, as we are only summing over y != x_t anyway
#   # therefore we are skipping over everything that's not in Q(mask, -))
#   scores = scores[..., model.graph.absorbing_state]
#   return (
#     (
#       dsigma
#       # TODO: unsure about exp / log
#       * (scores.exp() - ratio * scores + ratio * (ratio.log() - 1))
#       * nonpadded_mask
#       # we want Q(mask, mask) = 0 additionally
#       * nonabsorbed_mask
#       # summing (over the L) makes the loss treat scorenet output as logprobs
#       # (scores \approx \frac{p_t(y)}{p_t(x)})
#     ).sum(dim=-1)
#     / nonpadded_mask.sum()
#   ).mean()


def loss_dwdse(model, x0, t):
  absorbing_state = model.graph.absorbing_state
  sigma, dsigma = model.noise(t)
  x = model.graph.sample_transition(x0, sigma)
  score = model.scorenet(x, sigma)
  rel_ind = x == absorbing_state
  esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
  ratio = 1 / esigm1.expand_as(x)[rel_ind]
  other_ind = x0[rel_ind]
  # negative_term
  neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
  # positive term
  pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
  # constant term
  const = ratio * (ratio.log() - 1)
  entropy = torch.zeros(*x.shape, device=x.device)
  entropy[rel_ind] += pos_term - neg_term + const
  return (dsigma * entropy).sum(dim=-1).mean()
