import torch


# integral via single MC sample of t
def loss_dwdse(sedd, batch, t):
  sigma, dsigma = sedd.noise(t)

  # TODO: why is ratio = 1.0 / (exp(sigma) - 1) ?
  # for numerical stability
  esigm1 = torch.where(
    sigma < 0.5,
    torch.expm1(sigma),
    torch.exp(sigma) - 1,
  )  # (B,)
  ratio = (1.0 / esigm1).expand_as(batch)

  perturbed = sedd.graph.sample_transition(batch, sigma)
  # TODO: not sure if we really need it. embedding and transformer both have it masked
  nonpadded_mask = batch != sedd.scorenet.pad_idx
  absorbed_mask = perturbed == sedd.graph.absorbing_state
  score = sedd.scorenet(perturbed, sigma)
  # TODO: (Batch, seqLen, Vocabsize) -> (B, L) use MASK column. why?
  score = score[..., sedd.graph.absorbing_state]
  # what were doing here is skipping over everything that's not
  # in Q(mask, -)) - all other entries are -1 (or 0)
  # for the final entry, our original batch never contains a mask,
  # therefore Q(mask, mask) = 0 as it should be
  return (
    dsigma
    * (score - ratio * score.log() + ratio * (ratio.log() - 1))
    * nonpadded_mask
    * absorbed_mask
  ).sum() / nonpadded_mask.sum()
