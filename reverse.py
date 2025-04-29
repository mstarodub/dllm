import torch

import util


# returns exp(a*Q^absorb), with size [V,V]
def expm_absorbing(a, v):
  device = util.device()
  expm = torch.zeros((v, v), device=device)
  exp_neg = torch.exp(-a)
  expm.fill_diagonal_(exp_neg)
  # last row
  expm[v - 1, :-1] = 1 - exp_neg
  # bottom right
  expm[v - 1, v - 1] = 1.0
  return expm


def sample(model, steps, nbatches=1):
  device = util.device()
  vocab_absorbing_size = model.scorenet.output_layer.out_features
  xt = torch.full(
    (nbatches, model.scorenet.max_seq_len),
    model.graph.absorbing_state,
    device=device,
  )
  timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
  sigma_total = model.noise.noise_total(timesteps)
  for idx in range(steps):
    sigma_cur, sigma_next = sigma_total[idx], sigma_total[idx + 1]
    sigma_batch = sigma_cur.expand(nbatches, 1)

    # TODO: maybe we dont need 2 calls
    expm_fwd = expm_absorbing(sigma_cur - sigma_next, vocab_absorbing_size)
    expm_rev = expm_absorbing(sigma_next - sigma_cur, vocab_absorbing_size)

    scores = model.scorenet(xt, sigma_batch)

    probs = torch.einsum('ij,blj->bli', expm_fwd, scores) * expm_rev[xt]
    # we can sample from unnormalized; [B*L, V]
    probs_flat = probs.reshape(-1, vocab_absorbing_size)
    print('XXX', expm_rev)
    sampled = torch.multinomial(probs_flat, num_samples=1)
    xt = sampled.reshape(nbatches, model.scorenet.max_seq_len)

  return xt
