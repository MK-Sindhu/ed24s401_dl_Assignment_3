import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hid_dim, n_layers, cell, dropout):
        super().__init__()
        self.emb = nn.Embedding(inp_dim, emb_dim)
        RNN = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn  = RNN(emb_dim, hid_dim, n_layers,
                        batch_first=True,
                        dropout=dropout if n_layers>1 else 0)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        e = self.drop(self.emb(x))
        out, hidden = self.rnn(e)
        return out, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v    = nn.Linear(hid_dim,     1, bias=False)

    def forward(self, hidden, enc_out):
        # hidden: [B,H], enc_out: [B,S,H]
        B,S,H = enc_out.size()
        h_exp = hidden.unsqueeze(1).repeat(1,S,1)      # [B,S,H]
        energy= torch.tanh(self.attn(torch.cat([h_exp,enc_out],dim=2)))
        scores= self.v(energy).squeeze(2)               # [B,S]
        return F.softmax(scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, cell, dropout, use_attn=False):
        super().__init__()
        RNN = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.emb     = nn.Embedding(out_dim, emb_dim)
        self.use_attn= use_attn
        if use_attn:
            self.attn = Attention(hid_dim)

        in_dim = emb_dim + (hid_dim if use_attn else 0)
        self.rnn  = RNN(in_dim, hid_dim, n_layers,
                        batch_first=True,
                        dropout=dropout if n_layers>1 else 0)
        self.fc   = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok, hidden, enc_out=None):
        emb = self.drop(self.emb(tok).unsqueeze(1))  # [B,1,emb_dim]
        if self.use_attn:
            # get last layer hidden state
            h_last = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            w      = self.attn(h_last, enc_out)        # [B,S]
            ctx    = torch.bmm(w.unsqueeze(1), enc_out) # [B,1,H]
            r_in   = torch.cat([emb, ctx], dim=2)
        else:
            r_in = emb

        out, h2 = self.rnn(r_in, hidden)
        pred    = self.fc(out.squeeze(1))
        return pred, h2, (w if self.use_attn else None)

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, pad_idx, device):
        super().__init__()
        self.enc, self.dec = enc, dec
        self.pad_idx       = pad_idx
        self.device        = device

    def _init_dec_hidden(self, h_enc):
        # Project encoder hidden → decoder initial
        dec_layers = self.dec.rnn.num_layers
        if isinstance(h_enc, tuple):
            h,c = h_enc
            n,B,H = h.size()
            h0 = h.new_zeros(dec_layers,B,H)
            c0 = c.new_zeros(dec_layers,B,H)
            ncp = min(n, dec_layers)
            h0[-ncp:], c0[-ncp:] = h[-ncp:], c[-ncp:]
            return (h0,c0)
        else:
            h = h_enc
            n,B,H = h.size()
            h0 = h.new_zeros(dec_layers,B,H)
            ncp = min(n, dec_layers)
            h0[-ncp:] = h[-ncp:]
            return h0

    def forward(self, src, tgt, teacher_forcing=0.5):
        B,T = tgt.shape
        V   = self.dec.fc.out_features
        outputs = torch.zeros(B,T,V, device=self.device)
        enc_out, enc_h = self.enc(src)
        dec_h   = self._init_dec_hidden(enc_h)
        inp     = tgt[:,0]
        for t in range(1,T):
            pred, dec_h, _ = self.dec(inp, dec_h, enc_out if self.dec.use_attn else None)
            outputs[:,t] = pred
            top1 = pred.argmax(1)
            inp  = tgt[:,t] if torch.rand(1).item()<teacher_forcing else top1
        return outputs
