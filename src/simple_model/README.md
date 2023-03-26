# Simple model
### Trained on `small_train.parquet` which is the same as the first 10 parquets from rain.parquet`
Run `simple_model.py` to predict on `test.parquet`, the first time the script will attempt to find pretrained models in `pickles`. If none are found, it will train a new set of TF-IDF, SVD and logistic regression models.  

|     | Precision | Recall | f1-score | support |
| --- | --------- | ------ | -------- | ------- |
| False | 94% | 92% | 93% | 420653 |
| True | 83% | 87% | 85% | 180734 |
|  |  |  |  |  |
| Accuracy |  |  | 91% | 601387 |
| Macro avg. | 89% | 90% | 89% | 601387 |
| Weighted avg. | 91% | 91% | 91% | 601387 |
