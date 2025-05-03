import torch
from tqdm.auto import trange

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


# \tau-leaping tweedie denoising
@torch.no_grad()
def sample(model, steps, nbatches):
  device = util.device()
  model.scorenet.eval()
  vocab_absorbing_size = model.scorenet.output_layer.out_features
  xt = torch.full(
    (nbatches, model.scorenet.max_seq_len),
    model.graph.absorbing_state,
    device=device,
  )
  timesteps = torch.linspace(1.0, model.noise.eps, steps + 1, device=device)
  sigma_total = model.noise.noise_total(timesteps)
  for idx in trange(steps):
    # time flows from 1 to 0 in the reverse process
    sigma_cur, sigma_next = sigma_total[idx], sigma_total[idx + 1]
    sigma_batch = sigma_cur.expand(nbatches, 1)

    expm_fwd = expm_absorbing(sigma_next - sigma_cur, vocab_absorbing_size)
    expm_rev = expm_absorbing(sigma_cur - sigma_next, vocab_absorbing_size)

    # TODO: unsure (logprob)
    scores = model.scorenet(xt, sigma_batch).exp()

    probs = torch.einsum('ij,blj->bli', expm_fwd, scores) * expm_rev[xt]
    # we can sample from unnormalized; [B*L, V]
    # TODO: this still has weird values
    probs_flat = torch.nan_to_num(
      probs.reshape(-1, vocab_absorbing_size).clamp(min=1e-10, max=1e10), nan=0.0
    )
    sampled = torch.multinomial(probs_flat, num_samples=1)
    xt = sampled.reshape(nbatches, model.scorenet.max_seq_len)

  return xt


def sample_log(model, steps, log_extra=dict()):
  x = sample(model, steps, nbatches=1).squeeze(dim=0)
  util.log({'sample': model.tokenizer.decode(x), **log_extra})
