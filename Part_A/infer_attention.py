# Part_A/infer_attention.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import (TransliterationDataset, collate_fn,
                   Seq2Seq, Encoder, Decoder,
                   SRC_VOCAB, TGT_VOCAB, PAD_IDX)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'lexicons'
TEST_FILE= os.path.join(DATA_DIR,'hi.translit.sampled.test.tsv')

# re‐instantiate attention model (same HP as train_attention.py)
enc   = Encoder(len(SRC_VOCAB), 64, 256, 1, 'LSTM', 0.30)
dec   = Decoder(len(TGT_VOCAB), 64, 256, 1, 'LSTM', 0.30, use_attn=True)
model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)
model.load_state_dict(torch.load('models/best_attention_model2.pt'))
model.eval()

test_ds = TransliterationDataset(TEST_FILE, SRC_VOCAB, TGT_VOCAB)
test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

inv_src = {v:k for k,v in SRC_VOCAB.items()}
inv_tgt = {v:k for k,v in TGT_VOCAB.items()}

def decode(toks, inv):
    res=[]
    for idx in toks:
        ch=inv.get(idx.item(),'')
        if ch=='<eos>': break
        if ch not in ['<pad>','<sos>']:
            res.append(ch)
    return ''.join(res)

rows=[]
with torch.no_grad():
    for src, tgt in test_ld:
        out = model(src.to(DEVICE), tgt.to(DEVICE), teacher_forcing=0.0)
        pred= out.argmax(2)[0]
        rows.append([ decode(src[0],inv_src),
                      decode(tgt[0],inv_tgt),
                      decode(pred,inv_tgt) ])
os.makedirs('predictions_attention', exist_ok=True)
pd.DataFrame(rows, columns=['input','reference','prediction']) \
  .to_csv('predictions_attention/test_predictions_attention.tsv',
          sep='\t', index=False)
