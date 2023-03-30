# Fake News Project
Group 1 - shared repository related to the Fake News Project in Data Science 2023.

## Pipeline
In order to reproduce our pipeline:
1. Run `download_data.sh` which downloads, extracts and does some error correction to the dataset.
2. Go through the steps in `preprocess.ipynb`.
3. Choose which script to run, it being one of either:
   * `simple_model/simple_model.py`
   * `complex_models/complex_model_A.py`
   * `complex_models/dnn_complex.py`
   * `complex_models/xgboost_complex.py`

#### Folder strucure
Here is an overview of our folder-structure:
* fakenewsproject/
  * README.md
  * meetings/
    * ...
  * simple_model/
    * simple_model.py
  * complex_models/
    * bigdnn_complex.py
    * xgboost_complex.py
    * dnn_complex.py
  * report/
    * preamble.tex
    * master.tex
    * report.pdf
  * notebooks/
    * ...
