from typing import List


class CharTokenizer:
  def __init__(self, alphabet, bos=None, eos=None):
    self.chars = alphabet

    # this is not nice from a theory standpoint. we could fill every input
    # with spaces instead of PADs, and that way it would be recognized by
    # the markov process as changeable states. via choosing a good dataset
    # we can still have variable answer sizes (the model may learn to insert spaces)
    self.pad_token = '[PAD]'
    self.bos_token = bos
    self.eos_token = eos
    self.vocab = [
      self.pad_token,
      *(t for t in (self.bos_token, self.eos_token) if t is not None),
      *self.chars,
    ]
    self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
    self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
    self.pad_idx = self.char_to_idx[self.pad_token]
    self.bos_idx = self.char_to_idx.get(self.bos_token)
    self.eos_idx = self.char_to_idx.get(self.eos_token)
    print(self.idx_to_char.items())

  @property
  def vocab_size(self) -> int:
    return len(self.vocab)

  def encode(self, text: str) -> List[int]:
    def enc_tf(ch):
      idx = self.char_to_idx.get(ch)
      if idx is None:
        raise ValueError(f'char {ch} is not in vocabulary')
      return idx

    return [enc_tf(char) for char in text]

  def decode(self, indices: List[int], skip_special_tokens=True) -> str:
    chars = []
    for idx in indices:
      char = self.idx_to_char.get(int(idx), '□')
      if char == self.pad_token:
        # ignore PAD if skipping
        if skip_special_tokens:
          continue
        # stop decoding
        else:
          break
      chars.append(char)
    return ''.join(chars)

  def encode_all(self, seqs):
    return [
      [t for t in [self.bos_idx] if t is not None]
      + self.encode(seq)
      + [t for t in [self.eos_idx] if t is not None]
      for seq in seqs
    ]
