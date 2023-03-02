import pandas as pd
import os
import requests
import subprocess
from definitions import *
def download_raw_data(url: str):
    if (url[-3:] != 'csv'):
        raise Exception("Url must lead to csv file")
    response = requests.get(url)
    if response.ok:
        df = pd.read_csv(url)
        if (len(df) == 0):
            raise Exception("Dataframe is empty (try examining url)")
        else:
            df.to_csv(os.path.join(DATA_PATH, 'data.csv'))
    else:
        raise Exception("Issue connecting to server")
    return

def import_data(sample=True, overwrite=False):
    if sample:
        sample_file_path = os.path.join(DATA_PATH, "data.csv")
        if (not os.path.isfile(sample_file_path)) or overwrite:
            download_raw_data(SAMPLE_URL)
        return pd.read_csv(sample_file_path)
    else:
        dataset_file_path = os.path.join(DATA_PATH, "dataset.parquet")
        if (not os.path.isfile(dataset_file_path)) or overwrite:
            process = subprocess.Popen(["download_data.sh"], shell=True)
            process.wait() # Wait will its done downloading stuff
            df = pd.read_csv("news_cleaned_2018_02_13.csv")
            df.to_parquet(dataset_file_path)

            os.remove("joined_news.csv.zip")
        




