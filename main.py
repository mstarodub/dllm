import util
import dataset
import tokenizer
from sedd import Sedd, Trainer

cf = util.Config(
  dict(
    epochs=5,
    batch_size=2,
    snapshot_freq=None,
    # TODO: need a proper test set for both
    eval_freq=None,
    log_freq=1000,
    sample_freq=1000,
    sample_steps=128,
    lr=1e-3,
    dropout_p=0.0,
    max_seq_len=None,
    embed_dim=16,
    time_embed_dim=64,
    num_heads=2,
    num_layers=1,
  )
)


def sentences_experiment():
  cf.max_seq_len = 20
  text_tokenizer = tokenizer.CharTokenizer(dataset.ascii_alphabet())
  model = Sedd(cf, text_tokenizer)

  texts = [
    'hello world',
    'hi!',
    'diffusion models',
    'generating text',
  ]

  loader = dataset.text_dataset(
    text_tokenizer,
    texts,
    batch_size=cf.batch_size,
    max_seq_len=cf.max_seq_len,
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
