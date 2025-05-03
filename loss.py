import torch


# integral via single MC sample of t
def loss_dwdse_branchless(model, x0, t):
  sigma, dsigma = model.noise(t)
  xt = model.graph.sample_transition(x0, sigma)
  # M is number of absorbed tokens
  absorbed_mask = xt == model.graph.absorbing_state
  # [B, 1] -> [B, L] -> [M]
  dsigma = dsigma.expand_as(x0)[absorbed_mask]
  # numerically stable torch.exp(sigma) - 1
  ratio = 1.0 / torch.expm1(sigma).expand_as(x0)[absorbed_mask]
  # [B, L, V] -> [M, V]; predicted logits over possible tokens at absorbed positions
  scores = model.scorenet(xt, sigma)[absorbed_mask]
  # [M, 1]; original token at that position
  target = x0[absorbed_mask].unsqueeze(1)
  # extract elements at target
  neg = ratio * torch.gather(scores, 1, target).squeeze(1)
  # :-1 because Q(mask, mask) = 0
  pos = scores[:, :-1].exp().sum(dim=1)
  normalizing = ratio * (ratio.log() - 1)
  # [M]
  entropy = pos - neg + normalizing
  # mean over B
  return (dsigma * entropy).sum() / x0.shape[0]
