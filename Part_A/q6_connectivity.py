# Part_A/q6_connectivity.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from torch.utils.data import DataLoader
from utils import (TransliterationDataset, collate_fn,
                   Encoder, Decoder, Seq2Seq,
                   SRC_VOCAB, TGT_VOCAB, PAD_IDX)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ────────────────────────────────────────────────────
# 1) Font setup
# ────────────────────────────────────────────────────
dev_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
dev_prop = fm.FontProperties(fname=dev_path)
dej_prop = fm.FontProperties(fname=fm.findfont("DejaVu Sans"))
mpl.rcParams['font.family'] = [dev_prop.get_name(), dej_prop.get_name()]
mpl.rcParams['font.size']   = 12

# ────────────────────────────────────────────────────
# 2) load model
# ────────────────────────────────────────────────────
enc   = Encoder(len(SRC_VOCAB), 64, 256, 1, 'LSTM', 0.30)
dec   = Decoder(len(TGT_VOCAB), 64, 256, 1, 'LSTM', 0.30, use_attn=True)
model = Seq2Seq(enc, dec, PAD_IDX, DEVICE).to(DEVICE)
model.load_state_dict(torch.load('models/best_attention_model2.pt'))

# ────────────────────────────────────────────────────
# 3) connectivity function
# ────────────────────────────────────────────────────
def compute_connectivity(model, src_tensor, max_out_len):
    was_train = model.training
    model.train()  # cudnn backward requires train mode

    enc_out, enc_h = model.encoder(src_tensor)
    enc_out = enc_out.detach().requires_grad_(True)
    dec_h   = model._init_decoder_hidden(enc_h)
    inp     = torch.full((1,), TGT_VOCAB['<sos>'], device=DEVICE)

    steps = []
    for _ in range(max_out_len):
        logits, dec_h, _ = model.decoder(inp, dec_h, enc_out)
        pred_idx         = logits.argmax(-1).item()
        chosen           = logits[0, pred_idx]

        enc_out.grad = None
        chosen.backward(retain_graph=True)
        grad_norm = enc_out.grad[0].norm(dim=-1).cpu().numpy()
        steps.append(grad_norm)

        inp = torch.tensor([pred_idx], device=DEVICE)

    model.train(was_train)
    return np.stack(steps, axis=0)

# ────────────────────────────────────────────────────
# 4) sample 9 examples
# ────────────────────────────────────────────────────
test_ds = TransliterationDataset('lexicons/hi.translit.sampled.test.tsv',
                                 SRC_VOCAB, TGT_VOCAB)
loader  = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
samples = []
for src, tgt in loader:
    samples.append((src.to(DEVICE), tgt.to(DEVICE)))
    if len(samples) == 9: break

# ────────────────────────────────────────────────────
# 5) plot 3×3 grid + shared colorbar
# ────────────────────────────────────────────────────
fig, axes = plt.subplots(3,3, figsize=(16,16), constrained_layout=True)
last_im = None

for i, (src, tgt) in enumerate(samples):
    src_idxs = [x for x in src[0].tolist() if x not in (SRC_VOCAB['<pad>'],SRC_VOCAB['<eos>'])]
    tgt_idxs = [x for x in tgt[0].tolist() if x not in (TGT_VOCAB['<pad>'],TGT_VOCAB['<sos>'],TGT_VOCAB['<eos>'])]
    conn     = compute_connectivity(model, src, max_out_len=len(tgt_idxs))
    mat      = conn[:len(tgt_idxs), :len(src_idxs)]

    ax = axes[i//3, i%3]
    im = ax.imshow(mat, cmap='PuBu', aspect='auto', vmin=0.0, vmax=mat.max())
    last_im = im

    ax.set_xticks(range(len(src_idxs)))
    ax.set_yticks(range(len(tgt_idxs)))
    ax.set_xticklabels([list(SRC_VOCAB.keys())[list(SRC_VOCAB.values()).index(x)] for x in src_idxs],
                       rotation=90, fontsize=10)
    ax.set_yticklabels([list(TGT_VOCAB.keys())[list(TGT_VOCAB.values()).index(x)] for x in tgt_idxs],
                       fontsize=10)
    ax.set_xlabel("Roman (Input)")
    ax.set_ylabel("देवनागरी (आउटपुट)")
    ax.set_title(f"Example {i+1}")

# one shared colorbar
cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
cbar.set_label("Gradient L₂‐norm", rotation=90)
fig.savefig("attention_connectivity_grid.png", dpi=300)
plt.show()
