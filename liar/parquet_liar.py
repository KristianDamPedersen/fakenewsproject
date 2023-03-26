# Used to turn TSV into parquet and remove everything other than content and type

import pandas as pd

df = pd.read_csv("train.tsv", sep="\t", header=None)

df["type"] = "liar"

df = df[[2, "type"]]

df.columns = ["content", "type"]

df.to_parquet("liar.parquet")
