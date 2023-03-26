import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from custom_tokeniser import custom_tokenizer
import pickle
import tensorflow


def pass_fun(doc):
    return doc


try:
    print("loading pretrained svd and tfidf")
    tfidf = pickle.load(open("tfidf-dnn.pkl", "rb"))
    svd = pickle.load(open("svd-dnn.pkl", "rb"))
except:
    print("models not found, fitting new model")
    df = pd.read_parquet("small_train.parquet", columns=["tokens", "class"])
    X_train = df["tokens"]
    y_train = df["class"]

    print(y_train.value_counts())

    tfidf = TfidfVectorizer(
        max_features=4096, sublinear_tf=True, preprocessor=pass_fun, tokenizer=pass_fun
    )

    X_train = tfidf.fit_transform(X_train)
    print("saving tfidf model")
    pickle.dump(tfidf, open("pickles/tfidf-dnn.pkl", "wb"))

    svd = TruncatedSVD(n_components=386, random_state=42)
    X_train = svd.fit_transform(X_train)
    print("saving svd model")
    pickle.dump(svd, open("pickles/svd-dnn.pkl", "wb"))

df_test = pd.read_parquet("test.parquet")

X_test = df_test["tokens"]
y_test = df_test["class"]

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X_test = svd.transform(tfidf.transform(X_test))
y_pred = xgb.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(f"Accuracy {acc}")

print(classification_report(y_test, y_pred))
