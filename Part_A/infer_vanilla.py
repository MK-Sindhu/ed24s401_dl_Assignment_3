# Part_A/infer_vanilla.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import (TransliterationDataset, collate_fn,
                   Seq2Seq, Encoder, Decoder,
                   SRC_VOCAB, TGT_VOCAB, PAD_IDX)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR= 'lexicons'
TEST_FILE = os.path.join(DATA_DIR,'hi.translit.sampled.test.tsv')

# re‐instantiate the model
enc   = Encoder(len(SRC_VOCAB), 256, 256, 3, 'LSTM', 0.20)
dec   = Decoder(len(TGT_VOCAB), 256, 256, 2, 'LSTM', 0.20, use_attn=False)
model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)
model.load_state_dict(torch.load('models/best_model_vanilla.pt'))
model.eval()

# data loader
test_ds = TransliterationDataset(TEST_FILE, SRC_VOCAB, TGT_VOCAB)
test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# invert vocabs
inv_src = {v:k for k,v in SRC_VOCAB.items()}
inv_tgt = {v:k for k,v in TGT_VOCAB.items()}

def decode(toks, inv):
    res=[]
    for idx in toks:
        ch = inv.get(idx.item(), '')
        if ch=='<eos>': break
        if ch not in ['<pad>','<sos>']:
            res.append(ch)
    return ''.join(res)

# inference
rows=[]
with torch.no_grad():
    for src, tgt in test_ld:
        out = model(src.to(DEVICE), tgt.to(DEVICE), teacher_forcing=0.0)
        pred= out.argmax(2)[0]
        rows.append([ decode(src[0],inv_src),
                      decode(tgt[0],inv_tgt),
                      decode(pred,inv_tgt) ])

os.makedirs('predictions_vanilla', exist_ok=True)
pd.DataFrame(rows, columns=['input','reference','prediction']) \
  .to_csv('predictions_vanilla/test_predictions.tsv', sep='\t', index=False)
