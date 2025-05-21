# Part_A/train_attention.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import (TransliterationDataset, collate_fn,
                   Encoder, Decoder, Seq2Seq,
                   SRC_VOCAB, TGT_VOCAB, PAD_IDX,
                   compute_tok_acc)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'lexicons'
train_ds = TransliterationDataset(os.path.join(DATA_DIR,'hi.translit.sampled.train.tsv'),
                                  SRC_VOCAB, TGT_VOCAB)
dev_ds   = TransliterationDataset(os.path.join(DATA_DIR,'hi.translit.sampled.dev.tsv'),
                                  SRC_VOCAB, TGT_VOCAB)
test_ds  = TransliterationDataset(os.path.join(DATA_DIR,'hi.translit.sampled.test.tsv'),
                                  SRC_VOCAB, TGT_VOCAB)

train_ld = DataLoader(train_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn)
dev_ld   = DataLoader(dev_ds,   batch_size=128, shuffle=False, collate_fn=collate_fn)
test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False, collate_fn=collate_fn)

# ── Q5 example hp ────────────────────────────────────────────────────────────
EMB_DIM, HID_DIM     = 64, 256
ENC_LAYERS, DEC_LAYERS = 1, 1
CELL, DROPOUT        = 'LSTM', 0.30
LR, EPOCHS           = 1e-3, 20

enc   = Encoder(len(SRC_VOCAB), EMB_DIM, HID_DIM, ENC_LAYERS, CELL, DROPOUT)
dec   = Decoder(len(TGT_VOCAB), EMB_DIM, HID_DIM, DEC_LAYERS, CELL, DROPOUT, use_attn=True)
model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)

opt       = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

best_val_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    tr_l, tr_a = 0.0, 0.0
    for src, tgt in train_ld:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        opt.zero_grad()
        out = model(src, tgt, teacher_forcing=0.5)
        logits = out[:,1:].reshape(-1,out.size(-1))
        gold   = tgt[:,1:].reshape(-1)
        loss   = criterion(logits,gold)
        loss.backward(); opt.step()
        tr_l += loss.item()
        tr_a += compute_tok_acc(out[:,1:], tgt[:,1:])
    tr_l /= len(train_ld); tr_a /= len(train_ld)

    # validate
    model.eval()
    vl_l, vl_a = 0.0, 0.0
    with torch.no_grad():
        for src, tgt in dev_ld:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            out = model(src, tgt, teacher_forcing=0.0)
            logits = out[:,1:].reshape(-1,out.size(-1))
            gold   = tgt[:,1:].reshape(-1)
            loss   = criterion(logits,gold)
            vl_l += loss.item()
            vl_a += compute_tok_acc(out[:,1:], tgt[:,1:])
    vl_l /= len(dev_ld); vl_a /= len(dev_ld)

    print(f"Epoch {epoch} | TrainLoss {tr_l:.4f} TrainAcc {tr_a:.4f}"
          f" | ValLoss {vl_l:.4f} ValAcc {vl_a:.4f}")

    if vl_l < best_val_loss:
        best_val_loss = vl_l
        torch.save(model.state_dict(), 'models/best_attention_model2.pt')

# final test
model.load_state_dict(torch.load('models/best_attention_model2.pt'))
model.eval()
test_l, test_a = 0.0, 0.0
with torch.no_grad():
    for src, tgt in test_ld:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        out = model(src, tgt, teacher_forcing=0.0)
        logits = out[:,1:].reshape(-1,out.size(-1))
        gold   = tgt[:,1:].reshape(-1)
        test_l += criterion(logits,gold).item()
        test_a += compute_tok_acc(out[:,1:], tgt[:,1:])
test_l /= len(test_ld); test_a /= len(test_ld)
print(f"\nAttention‐Model Test Loss {test_l:.4f} Test Acc {test_a:.4f}")
