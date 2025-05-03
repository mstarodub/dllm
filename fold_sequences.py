import polars as pl
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import dataset

api_url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
timeout = 60
max_workers = 5
seq_start = '<s>'
seq_end = '</s>'
absorbing_tok = '□'
train_seqs = set(dataset.get_acyp_sequences())


def clean_sequence(sample):
  start = sample.find(seq_start)
  end = sample.find(seq_end, start)
  if start == -1 or end == -1:
    return ''
  core = sample[start + len(seq_start) : end]
  return core.replace(absorbing_tok, '')


def is_clean(sample):
  return (
    sample.startswith(seq_start)
    and sample.strip().endswith(seq_end)
    and sample.count(seq_start) == 1
    and sample.count(seq_end) == 1
  )


def masking_pct(sample):
  total = len(sample)
  if total == 0:
    return 0.0
  return sample.count(absorbing_tok) / total * 100


def fold_test(seq):
  try:
    r = requests.post(api_url, data=seq, timeout=timeout)
    return r.status_code == 200
  except requests.exceptions.RequestException:
    return False


def process_row(row):
  sample = row['sample']
  step = row['step']
  pct = round(masking_pct(sample), 1)
  clean_flag = is_clean(sample)
  seq = clean_sequence(sample)
  new_flag = seq not in train_seqs
  success = False
  if seq:
    success = fold_test(seq)
  return {
    'step': step,
    'sequence': seq,
    'maskingpct': pct,
    'clean': clean_flag,
    'successfulfold': success,
    'new': new_flag,
  }


if __name__ == '__main__':
  df = pl.read_csv('samples/proteins.csv')
  results = []
  with ThreadPoolExecutor(max_workers=max_workers) as exe:
    futures = {exe.submit(process_row, r): r for r in df.to_dicts()}
    for f in as_completed(futures):
      res = f.result()
      print(f'row {res["step"]} completed')
      results.append(f.result())
  out_df = pl.from_dicts(results).sort('step')
  out_df.write_csv('samples/fold_results.csv')
