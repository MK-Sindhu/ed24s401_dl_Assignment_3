# Part_A/train_vanilla.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import (TransliterationDataset, collate_fn,
                   Encoder, Decoder, Seq2Seq,
                   SRC_VOCAB, TGT_VOCAB, PAD_IDX,
                   compute_tok_acc)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR= 'lexicons'
train_ds= TransliterationDataset(os.path.join(DATA_DIR,'hi.translit.sampled.train.tsv'),
                                 SRC_VOCAB, TGT_VOCAB)
dev_ds  = TransliterationDataset(os.path.join(DATA_DIR,'hi.translit.sampled.dev.tsv'),
                                 SRC_VOCAB, TGT_VOCAB)
test_ds = TransliterationDataset(os.path.join(DATA_DIR,'hi.translit.sampled.test.tsv'),
                                 SRC_VOCAB, TGT_VOCAB)

train_ld= DataLoader(train_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn)
dev_ld  = DataLoader(dev_ds,   batch_size=128, shuffle=False, collate_fn=collate_fn)
test_ld = DataLoader(test_ds,  batch_size=128, shuffle=False, collate_fn=collate_fn)

# ── Q2 best hyperparams ───────────────────────────────────────────────────────
EMB_DIM, HID_DIM     = 256, 256
ENC_LAYERS, DEC_LAYERS = 3, 2
CELL, DROPOUT        = 'LSTM', 0.20
LR, EPOCHS           = 1e-3, 10

enc   = Encoder(len(SRC_VOCAB), EMB_DIM, HID_DIM, ENC_LAYERS, CELL, DROPOUT)
dec   = Decoder(len(TGT_VOCAB), EMB_DIM, HID_DIM, DEC_LAYERS, CELL, DROPOUT, use_attn=False)
model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)

opt       = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

best_val_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    # ── train ───────────────────────────────────────────────────────────────
    model.train()
    tot_loss, tot_acc = 0.0, 0.0
    for src, tgt in train_ld:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        opt.zero_grad()
        out    = model(src, tgt, teacher_forcing=0.5)
        logits = out[:,1:].reshape(-1,out.size(-1))
        gold   = tgt[:,1:].reshape(-1)
        loss   = criterion(logits, gold)
        loss.backward(); opt.step()
        tot_loss += loss.item()
        tot_acc  += compute_tok_acc(out[:,1:], tgt[:,1:])
    train_loss = tot_loss/len(train_ld)
    train_acc  = tot_acc/len(train_ld)

    # ── validate ────────────────────────────────────────────────────────────
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for src, tgt in dev_ld:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            out    = model(src, tgt, teacher_forcing=0.0)
            logits = out[:,1:].reshape(-1,out.size(-1))
            gold   = tgt[:,1:].reshape(-1)
            loss   = criterion(logits, gold)
            val_loss += loss.item()
            val_acc  += compute_tok_acc(out[:,1:], tgt[:,1:])
    val_loss /= len(dev_ld)
    val_acc  /= len(dev_ld)

    print(f"Epoch {epoch} | "
          f"TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} | "
          f"ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model_vanilla.pt')

# ── final test ───────────────────────────────────────────────────────────────
model.load_state_dict(torch.load('models/best_model_vanilla.pt'))
model.eval()
test_loss, test_acc = 0.0, 0.0
with torch.no_grad():
    for src, tgt in test_ld:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        out    = model(src, tgt, teacher_forcing=0.0)
        logits = out[:,1:].reshape(-1,out.size(-1))
        gold   = tgt[:,1:].reshape(-1)
        test_loss += criterion(logits,gold).item()
        test_acc  += compute_tok_acc(out[:,1:], tgt[:,1:])
test_loss /= len(test_ld)
test_acc  /= len(test_ld)
print(f"\nTest Loss {test_loss:.4f} Test Token Acc {test_acc:.4f}")
