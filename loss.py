import torch


# integral via single MC sample of t
def loss_dwdse(sedd, x0, t):
  sigma, dsigma = sedd.noise(t)

  esigm1 = torch.where(
    sigma < 0.5,
    torch.expm1(sigma),
    torch.exp(sigma) - 1,
  )
  # expand_as to make the sigma scalar equal for every token
  ratio = (1.0 / esigm1).expand_as(x0)

  xt = sedd.graph.sample_transition(x0, sigma)
  nonpadded_mask = x0 != sedd.scorenet.pad_idx
  nonabsorbed_mask = xt != sedd.graph.absorbing_state
  score = sedd.scorenet(xt, sigma)
  # (Batch, seqLen, Vocabsize) -> (B, L):
  # using the MASK column we get the last row - all other contributions
  # are 0, as we are only summing over y != x_t anyway
  # therefore we are skipping over everything that's not in Q(mask, -))
  score = score[..., sedd.graph.absorbing_state]
  return (
    (
      dsigma
      * (score - ratio * score.log() + ratio * (ratio.log() - 1))
      * nonpadded_mask
      # we want Q(mask, mask) = 0 additionally
      * nonabsorbed_mask
      # summing (over the L) makes the loss treat scorenet output as logprobs
      # (score \approx \frac{p_t(y)}{p_t(x)})
    ).sum(dim=-1)
    / nonpadded_mask.sum()
  ).mean()
