import pandas as pd
import pickle
import numpy as np
from lib.pass_fun import pass_fun
from lib.custom_tokeniser import custom_tokenizer
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model


def load_models():
    models = {}
    
    models['tfidf_2048'] = pickle.load(open("data/tfidf-2048.pkl", "rb"))
    models['svd_256'] = pickle.load(open("data/svd-256.pkl", "rb"))
    models['logreg'] = pickle.load(open("data/logreg.pkl", "rb"))
    models['tfidf_4096'] = pickle.load(open('data/tfidf-4096.pkl', "rb"))
    models['svd_384'] = pickle.load(open("data/svd-384.pkl", "rb"))
    models['dnn'] = load_model("data/smollboi2")
    models['bigchungus'] = load_model("data/biggus_chungus")
    models['xgb'] = XGBClassifier()
    models['xgb'].load_model("data/xgb.json")
    
    return models


def evaluate_pipelines(pipelines, models, X, y):
    for pipeline in pipelines:
        evaluate_pipeline(pipeline, models, X, y)



def evaluate_pipeline(pipeline, models, X, y):
    # Apply transformations
    for step in pipeline['transformations']:
        X = models[step].transform(X)

    # Predict and evaluate model
    model = models[pipeline['model']]
    if pipeline['name'] == "Big DNN":
        X = X.todense()
#    if pipeline['type'] == 'sklearn':
#        y_pred = model.predict(X)
#    elif pipeline['type'] == 'tensorflow':
#        y_pred = np.argmax(model.predict(X), axis=-1)
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)

    print(f"Results for {pipeline['name']}")
    print(classification_report(y, y_pred))


def run_pipelines(pipelines, models, X, y):
    for pipeline in pipelines:
        run_pipeline(pipeline, models, X, y)

models = load_models()
df = pd.read_parquet("liar/liar.parquet", engine='fastparquet')
X_tokens = df["tokens"]
y = np.array(df["class"])

pipelines = [
    {
        'name': 'Simple Model',
        'type': 'sklearn',
        'transformations': ['tfidf_2048', 'svd_256'],
        'model': 'logreg'
    },
    {
        'name': 'Big DNN',
        'type': 'tensorflow',
        'transformations': ['tfidf_4096'],
        'model': 'bigchungus'
    },
    {
        'name': 'Small DNN',
        'type': 'tensorflow',
        'transformations': ['tfidf_4096', 'svd_384'],
        'model': 'dnn'
    },
    {
        'name': 'XGBoost',
        'type': 'sklearn',
        'transformations': ['tfidf_4096', 'svd_384'],
        'model': 'xgb'
    }
]

evaluate_pipelines(pipelines, models, X_tokens, y)


def transform(chunk, modl):
    return svd.transform(tfidf.transform(chunk))

