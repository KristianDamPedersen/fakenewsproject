from src.preprocess import preprocess as pp
from definitions import DATA_PATH
import pytest
import os

VALID_URL = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"

def test_download_data_creates_csv_file():
    file_exists_initially = os.path.isfile(os.path.join(DATA_PATH, 'data.csv'))
    if file_exists_initially:
        os.remove(os.path.join(DATA_PATH, 'data.csv'))
    pp.download_raw_data(url=VALID_URL)
    file_exists = os.path.isfile(os.path.join(DATA_PATH, 'data.csv'))
    assert file_exists

def test_download_throws_error_on_invalid_url():
    with pytest.raises(Exception):
        pp.download_raw_data(url="bla bla bla")

# Returns error on empty df
def test_download_throws_error_empty_df():
    with pytest.raises(Exception):
        pp.download_raw_data(url='https://rollbar.com/blog/throwing-exceptions-in-python/#')

def test_cleanup():
    os.remove(os.path.join(DATA_PATH, 'data.csv'))

