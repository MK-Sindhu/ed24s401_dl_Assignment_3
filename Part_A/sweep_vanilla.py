# Part_A/sweep_vanilla.py
import os
import wandb
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
TRAIN_FILE = os.path.join(DATA_DIR,'hi.translit.sampled.train.tsv')
DEV_FILE   = os.path.join(DATA_DIR,'hi.translit.sampled.dev.tsv')

sweep_cfg = {
    'method':'bayes',
    'metric':{'name':'val_loss','goal':'minimize'},
    'parameters':{
      'emb_dim':   {'values':[16,32,64,256]},
      'hid_dim':   {'values':[16,32,64,256]},
      'enc_layers':{'values':[1,2,3]},
      'dec_layers':{'values':[1,2,3]},
      'cell':      {'values':['RNN','GRU','LSTM']},
      'dropout':   {'values':[0.2,0.3]},
      'lr':        {'value':1e-3},
      'batch_size':{'value':128}
    }
}

def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    if train: model.train()
    else:     model.eval()
    tot_loss, tot_acc = 0.0, 0.0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        if train: opt.zero_grad()
        out    = model(src, tgt, teacher_forcing=0.5 if train else 0.0)
        logits = out[:,1:].reshape(-1,out.size(-1))
        gold   = tgt[:,1:].reshape(-1)
        loss   = crit(logits, gold)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        tot_loss += loss.item()
        tot_acc  += compute_tok_acc(out[:,1:], tgt[:,1:])
    return tot_loss/len(loader), tot_acc/len(loader)

def sweep_run():
    wandb.init()
    cfg = wandb.config

    train_ds= TransliterationDataset(TRAIN_FILE, SRC_VOCAB, TGT_VOCAB)
    dev_ds  = TransliterationDataset(DEV_FILE, SRC_VOCAB, TGT_VOCAB)
    tr_ld   = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fn)
    dv_ld   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    enc = Encoder(len(SRC_VOCAB), cfg.emb_dim, cfg.hid_dim, cfg.enc_layers, cfg.cell, cfg.dropout)
    dec = Decoder(len(TGT_VOCAB), cfg.emb_dim, cfg.hid_dim, cfg.dec_layers, cfg.cell, cfg.dropout, use_attn=False)
    model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit= nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_vl = float('inf')
    for epoch in range(1,11):
        tr_l, tr_a = run_epoch(model, tr_ld, crit, opt)
        vl_l, vl_a = run_epoch(model, dv_ld, crit, None)
        wandb.log({'epoch':epoch,'train_loss':tr_l,'train_acc':tr_a,'val_loss':vl_l,'val_acc':vl_a})
        if vl_l < best_vl:
            best_vl = vl_l
            torch.save(model.state_dict(), 'models/best_model_vanilla.pt')
            wandb.save('models/best_model_vanilla.pt')

if __name__=='__main__':
    sweep_id = wandb.sweep(sweep_cfg, project='assignment3_sweep_vanilla')
    wandb.agent(sweep_id, function=sweep_run)
