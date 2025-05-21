import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def build_vocab(seqs, specials=['<pad>','<sos>','<eos>']):
    chars = set(''.join(seqs))
    idx = {tok:i for i,tok in enumerate(specials)}
    for c in sorted(chars):
        idx[c] = len(idx)
    return idx

class TransliterationDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab, max_len=32):
        df = pd.read_csv(path, sep='\t', names=['dev','rom','_']).dropna()
        self.pairs      = df[['rom','dev']].values.tolist()
        self.src_vocab  = src_vocab
        self.tgt_vocab  = tgt_vocab
        self.max_len    = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rom, dev = self.pairs[i]
        src_ids = [ self.src_vocab[c] for c in rom ][:self.max_len] + [self.src_vocab['<eos>']]
        tgt_ids = [ self.tgt_vocab['<sos>'] ] \
                + [ self.tgt_vocab[c] for c in dev ][:self.max_len] \
                + [ self.tgt_vocab['<eos>'] ]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_pad, tgt_pad
