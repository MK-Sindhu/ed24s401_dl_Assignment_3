import pandas as pd
from IPython.display import display, HTML

df = pd.read_csv('Part_A/predictions_vanilla/test_predictions.tsv', sep='\t')
display(HTML(df.sample(30, random_state=123).to_html(index=False)))
