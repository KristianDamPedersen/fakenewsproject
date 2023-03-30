# This is step 2
import sys
from lib.custom_tokeniser import custom_tokenizer
from multiprocessing import Pool
import pandas as pd
import glob


# This function takes a parquet dataset and tokenises it according to the custom tokeniser we've implemented
def tokenise_parquet(parquet_file):
    print(f"Reading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    df["tokens"] = df["content"].apply(custom_tokenizer)
    df.to_parquet(parquet_file)
    print(f"Finished writing {parquet_file}")


# Take one command argument which is the parquet dataset and apply the function above to it
if __name__ == "__main__":
    parquet_dir = sys.argv[1]
    parquet_files = glob.glob(parquet_dir + "/*.parquet")
    p = Pool(8)
    p.map(tokenise_parquet, parquet_files)
