# Part_A/utils.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# ────────────────────────────────────────────────────
# 1) Vocabulary builder & initial vocab load from train
# ────────────────────────────────────────────────────
def build_vocab(seqs, specials=['<pad>','<sos>','<eos>']):
    chars = set(''.join(seqs))
    idx   = {tok:i for i,tok in enumerate(specials)}
    for c in sorted(chars):
        idx[c] = len(idx)
    return idx

# read train split to construct vocabs
_train_df   = pd.read_csv('lexicons/hi.translit.sampled.train.tsv',
                          sep='\t', names=['dev','rom','_']).dropna()
SRC_VOCAB   = build_vocab(_train_df['rom'])
TGT_VOCAB   = build_vocab(_train_df['dev'])
PAD_IDX     = TGT_VOCAB['<pad>']

# ────────────────────────────────────────────────────
# 2) Dataset + collate
# ────────────────────────────────────────────────────
class TransliterationDataset(Dataset):
    def __init__(self, path, src_map, tgt_map, max_len=32):
        df = pd.read_csv(path, sep='\t', names=['dev','rom','_']).dropna()
        self.pairs   = df[['rom','dev']].values.tolist()
        self.src_map = src_map
        self.tgt_map = tgt_map
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rom, dev = self.pairs[i]
        src = [self.src_map[c] for c in rom][:self.max_len] + [self.src_map['<eos>']]
        tgt = [self.tgt_map['<sos>']] + [self.tgt_map[c] for c in dev][:self.max_len] + [self.tgt_map['<eos>']]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=SRC_VOCAB['<pad>'])
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=TGT_VOCAB['<pad>'])
    return src_pad, tgt_pad

# ────────────────────────────────────────────────────
# 3) Encoder / Attention / Decoder / Seq2Seq
# ────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hid_dim, n_layers, rnn_type, dropout):
        super().__init__()
        self.emb = nn.Embedding(inp_dim, emb_dim)
        RNN      = {'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[rnn_type]
        self.rnn = RNN(emb_dim, hid_dim, n_layers,
                       batch_first=True,
                       dropout=dropout if n_layers>1 else 0)
        self.drop= nn.Dropout(dropout)

    def forward(self, x):
        e = self.drop(self.emb(x))
        out, h = self.rnn(e)
        return out, h

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.energy = nn.Linear(hid_dim*2, hid_dim)
        self.v      = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_state, enc_out):
        # dec_state: [B,H], enc_out: [B,S,H]
        B,S,H    = enc_out.size()
        dec_rep  = dec_state.unsqueeze(1).repeat(1,S,1)        # [B,S,H]
        e        = torch.tanh(self.energy(torch.cat([dec_rep, enc_out], dim=2)))  # [B,S,H]
        scores   = self.v(e).squeeze(2)                       # [B,S]
        return F.softmax(scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, rnn_type, dropout, use_attn=False):
        super().__init__()
        self.emb     = nn.Embedding(out_dim, emb_dim)
        self.use_attn= use_attn
        if use_attn:
            self.attn = Attention(hid_dim)
        in_dim = emb_dim + (hid_dim if use_attn else 0)
        RNN      = {'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[rnn_type]
        self.rnn  = RNN(in_dim, hid_dim, n_layers,
                        batch_first=True,
                        dropout=dropout if n_layers>1 else 0)
        self.fc   = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok, hidden, enc_out=None):
        # tok: [B]
        emb = self.drop(self.emb(tok).unsqueeze(1))  # [B,1,E]
        if self.use_attn:
            h_last = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            w      = self.attn(h_last, enc_out)       # [B,S]
            context= torch.bmm(w.unsqueeze(1), enc_out)  # [B,1,H]
            r_in   = torch.cat([emb, context], dim=2)
        else:
            r_in = emb
        out, h2 = self.rnn(r_in, hidden)
        pred    = self.fc(out.squeeze(1))
        return pred, h2, (w if self.use_attn else None)

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, pad_idx, device):
        super().__init__()
        self.encoder= enc
        self.decoder= dec
        self.pad_idx= pad_idx
        self.device = device

    def _init_decoder_hidden(self, enc_h):
        dec_n = self.decoder.rnn.num_layers
        if isinstance(enc_h, tuple):
            h,c    = enc_h
            h0     = h.new_zeros(dec_n, *h.shape[1:])
            c0     = c.new_zeros(dec_n, *c.shape[1:])
            n_copy = min(h.size(0), dec_n)
            h0[-n_copy:], c0[-n_copy:] = h[-n_copy:], c[-n_copy:]
            return (h0, c0)
        else:
            h0     = enc_h.new_zeros(dec_n, *enc_h.shape[1:])
            n_copy = min(enc_h.size(0), dec_n)
            h0[-n_copy:] = enc_h[-n_copy:]
            return h0

    def forward(self, src, tgt, teacher_forcing=0.5):
        B,T = tgt.size(); V = self.decoder.fc.out_features
        outputs= torch.zeros(B,T,V, device=self.device)
        enc_out, enc_h = self.encoder(src)
        dec_h          = self._init_decoder_hidden(enc_h)
        inp            = tgt[:,0]
        for t in range(1, T):
            logits, dec_h, _ = self.decoder(inp, dec_h,
                                            enc_out if self.decoder.use_attn else None)
            outputs[:,t] = logits
            top1         = logits.argmax(1)
            inp          = tgt[:,t] if torch.rand(1).item() < teacher_forcing else top1
        return outputs

# ────────────────────────────────────────────────────
# 4) Metrics
# ────────────────────────────────────────────────────
def compute_tok_acc(preds, tgts):
    with torch.no_grad():
        tok  = preds.argmax(dim=2)
        mask = tgts != PAD_IDX
        return ((tok==tgts)&mask).sum().float()/mask.sum().float()

def compute_seq_acc(preds, tgts):
    with torch.no_grad():
        tok   = preds.argmax(dim=2)
        mask  = (tok==tgts)|(tgts==PAD_IDX)
        correct = mask.all(dim=1).sum().item()
        total   = preds.size(0)
        return correct, total
