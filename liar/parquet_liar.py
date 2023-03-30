# Used to turn TSV into parquet and remove everything other than content and type

import pandas as pd
directory = 'liar/'

column_names = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']

train_df = pd.read_csv(directory+"train.tsv", sep="\t", header=None, names = column_names)
test_df = pd.read_csv(directory+"test.tsv", sep="\t", header=None, names = column_names)
val_df = pd.read_csv(directory+"valid.tsv", sep="\t", header=None, names = column_names)

df = pd.concat([train_df, test_df, val_df], ignore_index=True)

df["type"] = "liar"

df = df[['label', "type", "statement"]]
df["class"] = df["label"] == "true"

#df.columns = ["content", "type"]

df.to_parquet(directory+"liar.parquet")
