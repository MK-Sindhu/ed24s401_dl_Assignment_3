import torch
import torch.nn.functional as F
from queue import PriorityQueue

def compute_acc(preds, tgts, pad_idx):
    with torch.no_grad():
        guess = preds.argmax(2)
        mask  = tgts != pad_idx
        corr  = (guess==tgts) & mask
        return corr.sum().float()/mask.sum().float()

def epoch_step(model, loader, optimizer, criterion, train=True, pad_idx=None):
    if train: model.train()
    else:     model.eval()

    tot_loss, tot_acc = 0.0, 0.0
    for src, tgt in loader:
        src, tgt = src.to(model.device), tgt.to(model.device)
        if train: optimizer.zero_grad()

        out = model(src, tgt, teacher_forcing=0.5 if train else 0.0)
        logits = out[:,1:].reshape(-1, out.size(-1))
        gold   = tgt[:,1:].reshape(-1)
        loss   = criterion(logits, gold)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        tot_loss += loss.item()
        tot_acc  += compute_acc(out[:,1:], tgt[:,1:], pad_idx)
    return tot_loss/len(loader), tot_acc/len(loader)


class BeamNode:
    def __init__(self, h, prev, tok, lp, length):
        self.h, self.prev, self.tok, self.lp, self.length = h, prev, tok, lp, length
    def score(self):
        return self.lp / self.length

def beam_decode(model, src, beam_k=1, max_len=32):
    """
    Greedy / beam search decoding. Returns list of token‐ids (excluding <sos>).
    """
    model.eval()
    with torch.no_grad():
        enc_out, h0 = model.enc(src)
        start = torch.tensor([model.dec.emb.padding_idx], device=src.device)  # actually <sos>
        root  = BeamNode(h0, None, start, 0.0, 1)
        pq = PriorityQueue(); pq.put((-root.score(), root))
        finished = []

        while not pq.empty():
            _, node = pq.get()
            if node.tok.item()==model.dec.emb.padding_idx and node.prev:
                finished.append((node.score(), node))
                if len(finished)>=beam_k: break

            pred, h1, _ = model.dec(node.tok, node.h, enc_out if model.dec.use_attn else None)
            logps      = F.log_softmax(pred, 1)
            topv, topi = logps.topk(beam_k)

            for i in range(beam_k):
                nt = topi[0][i].unsqueeze(0)
                nl = node.lp + topv[0][i].item()
                child = BeamNode(h1, node, nt, nl, node.length+1)
                pq.put((-child.score(), child))

        best = max(finished, key=lambda x: x[0])[1]
        seq  = []
        while best.prev:
            seq.append(best.tok.item())
            best = best.prev
        return seq[::-1]
