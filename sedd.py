class Sedd:
  def __init__(self, scorenet, graph, noise):
    self.scorenet = scorenet
    self.graph = graph
    self.noise = noise


# just for testing
if __name__ == '__main__':
  import torch
  import util
  import score
  import noise
  import eval_train
  import dataset
  import tokenizer
  from loss import loss_dwdse
  from graph import AbsorbingGraph

  tokenizer = tokenizer.CharTokenizer(dataset.ascii_alphabet())
  pad_idx = tokenizer.pad_idx
  absorbing_idx = tokenizer.vocab_size
  max_seq_len = 20
  device = util.device()

  scorenet = score.ScoreNet(
    # masking/absorbing token
    vocab_size=tokenizer.vocab_size + 1,
    embed_dim=16,
    time_embed_dim=64,
    num_heads=2,
    num_layers=1,
    max_seq_len=max_seq_len,
    pad_idx=pad_idx,
    dropout=0.0,
  ).to(device)
  scorenet.eval()

  noise_schedule = noise.LogLinearNoise()

  texts = [
    'hello world',
    'hi!',
  ]
  encoded_batch = [tokenizer.encode(text) for text in texts]

  def pad(rep):
    return rep + [pad_idx] * (max_seq_len - len(rep))

  padded_batch = torch.tensor(
    [pad(rep) for rep in encoded_batch],
    dtype=torch.long,
    device=device,
  )

  t_batch = torch.rand(padded_batch.shape[0], device=device)
  total_noise_batch, rate_noise_batch = noise_schedule(t_batch)
  print('noise rate', rate_noise_batch)
  print('noise total', total_noise_batch)

  with torch.no_grad():
    output_logits = scorenet(padded_batch, total_noise_batch)

  print(f'output logits shape: {output_logits.shape}')

  predicted_0 = torch.argmax(output_logits[0], dim=-1).tolist()
  decoded_text_0 = tokenizer.decode(predicted_0)
  print(f'decoded output: "{decoded_text_0}"')

  graph = AbsorbingGraph(tokenizer.vocab_size)

  # def loss_dwdse(scorenet, graph, noise, batch, t):
  model = Sedd(scorenet, graph, noise_schedule)
  print(loss_dwdse(model, padded_batch, torch.tensor([0.4, 0.7])))

  cf = util.Config(
    dict(
      epochs=5,
      batch_size=2,
      snapshot_freq=None,
      eval_freq=1,
      log_freq=1,
      sample_freq=1,
      sample_steps=128,
      lr=1e-3,
    )
  )
  tr = eval_train.Trainer(model, cf)

  from torch.utils.data import DataLoader, TensorDataset

  ds = TensorDataset(padded_batch)
  loader = DataLoader(ds, batch_size=cf.batch_size, shuffle=True)
  tr.train(loader, loader)

  import reverse

  reverse.sample_log(model, steps=4)
