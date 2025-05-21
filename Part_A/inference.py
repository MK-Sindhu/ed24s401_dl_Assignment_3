import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_vocab, TransliterationDataset, collate_fn
from model   import Encoder, Decoder, Seq2Seq
from utils   import beam_decode

def main():
    # paths & vocabs
    lex = "lexicons"
    train_f = os.path.join(lex,"hi.translit.sampled.train.tsv")
    test_f  = os.path.join(lex,"hi.translit.sampled.test.tsv")

    df_train = pd.read_csv(train_f, sep="\t", names=["dev","rom","_"]).dropna()
    SRC_VOCAB = build_vocab(df_train["rom"])
    TGT_VOCAB = build_vocab(df_train["dev"])
    PAD_IDX   = TGT_VOCAB["<pad>"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best model
    enc = Encoder(len(SRC_VOCAB), 256, 256, 3, "LSTM", 0.2)
    dec = Decoder(len(TGT_VOCAB), 256, 256, 2, "LSTM", 0.2, use_attn=False)
    model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # run inference
    ds = TransliterationDataset(test_f, SRC_VOCAB, TGT_VOCAB)
    ld = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    inv_src = {v:k for k,v in SRC_VOCAB.items()}
    records = []
    for src, _ in ld:
        src = src.to(device)
        seq_ids = beam_decode(model, src, beam_k=1, max_len=32)
        # decode to string
        pred = "".join([ list(TGT_VOCAB.keys())[list(TGT_VOCAB.values()).index(i)]
                         for i in seq_ids if i not in (PAD_IDX,) ])
        inp = "".join([inv_src[idx.item()]
                       for idx in src[0] if inv_src[idx.item()] not in ['<pad>','<eos>']])
        records.append((inp, pred))

    os.makedirs("predictions_vanilla", exist_ok=True)
    pd.DataFrame(records, columns=["src","pred"])\
      .to_csv("predictions_vanilla/test_predictions.tsv", sep="\t", index=False)

    print("✔ Predictions saved to predictions_vanilla/test_predictions.tsv")

if __name__=="__main__":
    main()
