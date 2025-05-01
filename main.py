import util
import dataset
import tokenizer
from sedd import Sedd, Trainer

cf = util.Config(
  dict(
    steps=50000,
    batch_size=256,
    snapshot_freq=None,
    # TODO: need a proper test set for both
    eval_freq=None,
    log_freq=1000,
    sample_freq=1000,
    # 1024 for proteins, 128 for language
    sample_steps=10,
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
  bos, eos = ('<s>', '</s>') if add_markers else (None, None)
  protein_tokenizer = tokenizer.CharTokenizer(
    dataset.protein_alphabet(), bos=bos, eos=eos
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


if __name__ == '__main__':
  sentences_experiment()
  # protein_experiment()
