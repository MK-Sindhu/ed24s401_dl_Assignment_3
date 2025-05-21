import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import build_vocab, TransliterationDataset, collate_fn
from model   import Encoder, Decoder, Seq2Seq
from utils   import epoch_step

def main():
    # paths
    base       = os.path.dirname(__file__)
    lexicons   = os.path.join(base,"lexicons")
    train_f    = os.path.join(lexicons,"hi.translit.sampled.train.tsv")
    dev_f      = os.path.join(lexicons,"hi.translit.sampled.dev.tsv")

    # build vocabs
    df_train = pd.read_csv(train_f, sep="\t", names=["dev","rom","_"]).dropna()
    SRC_VOCAB = build_vocab(df_train["rom"])
    TGT_VOCAB = build_vocab(df_train["dev"])
    PAD_IDX   = TGT_VOCAB["<pad>"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets + loaders
    train_ds = TransliterationDataset(train_f, SRC_VOCAB, TGT_VOCAB)
    dev_ds   = TransliterationDataset(dev_f,   SRC_VOCAB, TGT_VOCAB)
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn)
    dev_ld   = DataLoader(dev_ds,   batch_size=128, shuffle=False, collate_fn=collate_fn)

    # build model (best params)
    enc = Encoder(len(SRC_VOCAB), 256, 256, 3, "LSTM", 0.2)
    dec = Decoder(len(TGT_VOCAB), 256, 256, 2, "LSTM", 0.2, use_attn=False)
    model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_acc = 0.0
    for epoch in range(1, 11):
        tr_l, tr_a = epoch_step(model, train_ld, optimizer, criterion, train=True,  pad_idx=PAD_IDX)
        vl_l, vl_a = epoch_step(model, dev_ld,   optimizer, criterion, train=False, pad_idx=PAD_IDX)
        print(f"Epoch {epoch:2d} | train_loss={tr_l:.4f} train_acc={tr_a:.4f} | val_loss={vl_l:.4f} val_acc={vl_a:.4f}")
        if vl_a > best_val_acc:
            best_val_acc = vl_a
            torch.save(model.state_dict(), "best_model.pt")

    print(f"✔ Saved best_model.pt @ val_acc={best_val_acc:.4f}")

if __name__ == "__main__":
    main()
