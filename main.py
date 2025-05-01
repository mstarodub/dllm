import util
import dataset
import tokenizer
from sedd import Sedd, Trainer

cf = util.Config(
  dict(
    steps=50000,
    batch_size=128,
    snapshot_freq=None,
    # TODO: need a proper test set for both
    eval_freq=None,
    log_freq=1000,
    sample_freq=1000,
    # 1024 for proteins, 128 for language
    sample_steps=128,
    lr=3e-4,
    warmup=2500,
    grad_clip=1.0,
    dropout_p=0.1,
    max_seq_len=None,
    # 768
    embed_dim=8,
    # 128
    time_embed_dim=8,
    # 8 for protein, 12 for langauge
    num_heads=2,
    num_layers=2,
    # 1024 for langauge, 128 for proteins (None would be saner)
    block_size=None,
  )
)


def sentences_experiment():
  cf.max_seq_len = 11
  text_tokenizer = tokenizer.CharTokenizer(dataset.ascii_alphabet())
  model = Sedd(cf, text_tokenizer)

  texts = [
    # 'abcdefghijk',
    'aaaaaaaaaaa',
    # 'a',
  ]

  loader = dataset.text_dataset(
    text_tokenizer,
    texts,
    batch_size=cf.batch_size,
    max_seq_len=cf.max_seq_len,
    block_size=cf.block_size,
  )
  trainer = Trainer(model, cf, data_train=loader, data_test=loader)
  trainer.train()


def protein_experiment(add_markers=True):
  cf.max_seq_len = 127
  protein_tokenizer = tokenizer.CharTokenizer(
    dataset.protein_alphabet(), add_special_tokens=add_markers
  )
  model = Sedd(cf, protein_tokenizer)

  loader = dataset.acyp_dataset(
    protein_tokenizer,
    batch_size=cf.batch_size,
    max_seq_len=cf.max_seq_len,
    # TODO
    block_size=None,
  )
  trainer = Trainer(
    model,
    cf,
    data_train=loader,
    data_test=loader,
    checkpoint_dir='checkpoints/proteins',
  )
  trainer.train()


def gpt2_experiment():
  cf.block_size = 1024
  cf.max_seq_len = cf.block_size
  cf.embed_dim = 768
  cf.time_embed_dim = 128
  cf.num_heads = 12
  cf.num_layers = 12

  gpt_tokenizer = tokenizer.GPTTokenizer()
  model = Sedd(cf, gpt_tokenizer)

  train_loader, test_loader = dataset.gpt2_dataset(
    gpt_tokenizer,
    batch_size=cf.batch_size,
    block_size=cf.block_size,
  )

  trainer = Trainer(
    model,
    cf,
    data_train=train_loader,
    data_test=test_loader,
    checkpoint_dir='checkpoints/gpt2',
  )
  trainer.train()


if __name__ == '__main__':
  # util.grad_debug()
  util.settings()

  # sentences_experiment()
  # protein_experiment()
  gpt2_experiment()
