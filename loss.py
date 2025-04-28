import torch


def loss_dwdse(scorenet, graph, noise, batch, t):
  sigma, dsigma = noise(t)

  # TODO: ratio = 1.0 / (exp(sigma) - 1), but we do something for
  # extra numerical stability
  esigm1 = torch.where(
    sigma < 0.5,
    torch.expm1(sigma),
    torch.exp(sigma) - 1,
  )  # (B,)
  ratio = (1.0 / esigm1).unsqueeze(1).expand_as(batch)

  # TODO: all of this is so fucking ugly
  # sigma: (B,)  → (B,1) so it broadcasts along sequence length
  sigma = sigma.unsqueeze(1)
  dsigma = dsigma.unsqueeze(1)

  perturbed = graph.sample_transition(batch, sigma)
  # TODO: not sure if we really need it. embedding and transformer both have it masked
  nonpadded_mask = batch != scorenet.pad_idx
  absorbed_mask = perturbed == graph.absorbing_state
  # (Batch, seqLen, Vocabsize)
  score = scorenet(perturbed, sigma)
  # (B, L)  use MASK column
  score = score[..., graph.absorbing_state]
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
