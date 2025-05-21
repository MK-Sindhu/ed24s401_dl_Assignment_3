# Part_A/compare_corrections.py

import os
import pandas as pd

# 1) Paths to your existing predictions
vanilla_path  = 'predictions_vanilla/test_predictions.tsv'
attention_path= 'predictions_attention/test_predictions_attention.tsv'

# 2) Load them (they have no header row, so we supply names)
vanilla_df = pd.read_csv(vanilla_path,
                         sep='\t',
                         header=None,
                         names=['input','reference','vanilla_pred'])
attn_df    = pd.read_csv(attention_path,
                         sep='\t',
                         header=None,
                         names=['input','reference','attn_pred'])

# 3) Merge on both columns
df = vanilla_df.merge(attn_df, on=['input','reference'])

# 4) Filter: vanilla wrong & attention correct
fixed = df[
    (df['vanilla_pred'] != df['reference']) &
    (df['attn_pred']    == df['reference'])
]

# 5) Print summary
print(f"Found {len(fixed)} cases where attention corrected the vanilla model.\n")
print(fixed[['input','reference','vanilla_pred','attn_pred']]
      .head(20)
      .to_string(index=False))

# 6) Save full list to a new folder
out_dir = 'predictions_comparison'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fixed_cases.tsv')
fixed.to_csv(out_path, sep='\t', index=False)

print(f"\nAll fixed cases saved to → {out_path}")
