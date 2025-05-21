import os
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import build_vocab, TransliterationDataset, collate_fn
from model   import Encoder, Decoder, Seq2Seq
from utils   import epoch_step


# ————————————————————————————————
# 1) Define your paths
LEX_DIR     = "lexicons"
TRAIN_FILE  = os.path.join(LEX_DIR, "hi.translit.sampled.train.tsv")
DEV_FILE    = os.path.join(LEX_DIR, "hi.translit.sampled.dev.tsv")

# 2) Build vocabs
df_train    = pd.read_csv(TRAIN_FILE, sep="\t", names=["dev","rom","_"]).dropna()
SRC_VOCAB   = build_vocab(df_train["rom"])
TGT_VOCAB   = build_vocab(df_train["dev"])
PAD_IDX     = TGT_VOCAB["<pad>"]

# 3) Device
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# — now you can define your sweep_cfg and sweep_run() exactly as you had it —




# --- Sweep config (Q2) ---
sweep_cfg = {
  "method":"bayes",
  "metric":{"name":"val_loss","goal":"minimize"},
  "parameters":{
    "emb_dim":   {"values":[16,32,64,256]},
    "hid_dim":   {"values":[16,32,64,256]},
    "enc_layers":{"values":[1,2,3]},
    "dec_layers":{"values":[1,2,3]},
    "cell":      {"values":["RNN","GRU","LSTM"]},
    "dropout":   {"values":[0.2,0.3]},
    "beam_k":    {"values":[1,3,5]},
    "lr":        {"value":1e-3},
    "batch_size":{"value":128}
  }
}

def sweep_run():
    wandb.init()
    cfg = wandb.config

    # data
    train_ds = TransliterationDataset("lexicons/hi.translit.sampled.train.tsv", 
                                      SRC_VOCAB, TGT_VOCAB)
    dev_ds   = TransliterationDataset("lexicons/hi.translit.sampled.dev.tsv", 
                                      SRC_VOCAB, TGT_VOCAB)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fn)
    dev_ld   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    enc = Encoder(len(SRC_VOCAB), cfg.emb_dim, cfg.hid_dim, cfg.enc_layers, cfg.cell, cfg.dropout)
    dec = Decoder(len(TGT_VOCAB), cfg.emb_dim, cfg.hid_dim, cfg.dec_layers, cfg.cell, cfg.dropout, use_attn=False)
    model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(1,11):
        tr_l, tr_a = epoch_step(model, train_ld, optimizer, criterion, train=True,  pad_idx=PAD_IDX)
        vl_l, vl_a = epoch_step(model, dev_ld,   optimizer, criterion, train=False, pad_idx=PAD_IDX)
        wandb.log({
            "epoch":epoch,
            "train_loss":tr_l, "train_acc":tr_a,
            "val_loss":vl_l,   "val_acc":vl_a
        })

if __name__=="__main__":
    sweep_id = wandb.sweep(sweep_cfg, project="assignment3_sweep")
    wandb.agent(sweep_id, function=sweep_run)
