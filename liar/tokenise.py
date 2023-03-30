# This is step 2
from lib.custom_tokeniser import custom_tokenizer
import pandas as pd


def tokenise_parquet(parquet_file):
    print(f"Reading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    df["tokens"] = df["statement"].apply(custom_tokenizer)
    df.to_parquet(parquet_file)
    print(f"Finished writing {parquet_file}")


parquet_file = "liar.parquet"

tokenise_parquet(parquet_file)
