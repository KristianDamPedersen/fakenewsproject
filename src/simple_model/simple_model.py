import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from custom_tokeniser import custom_tokenizer
from sklearn.metrics import classification_report
import pickle

# Does nothing, is used to pass through the custom tokenizer to the tfidf vectorizer.
def pass_fun(doc):
    return doc

# Load either the pretrained models or train new models
try:
    print("loading pretrained models")
    tfidf = pickle.load(open("data/tfidf-2048.pkl", "rb"))
    svd = pickle.load(open("data/svd-256.pkl", "rb"))
    lr = pickle.load(open("data/logreg.pkl", "rb"))
except:
    print("models not found, fitting new model")
    df = pd.read_parquet("data/small_train.parquet", columns=["tokens", "class"], engine="fastparquet")
    X_train = df["tokens"]
    y_train = df["class"]

    print(y_train.value_counts())

    tfidf = TfidfVectorizer(
        max_features=2048, sublinear_tf=True, preprocessor=pass_fun, tokenizer=pass_fun
    )

    print("fitting TF-IDF")
    X_train = tfidf.fit_transform(X_train)
    print("saving tfidf model")
    pickle.dump(tfidf, open("data/tfidf-2048.pkl", "wb"))

    print("fitting SVD")
    svd = TruncatedSVD(n_components=256, random_state=42)
    X_train = svd.fit_transform(X_train)
    print("saving svd model")
    pickle.dump(svd, open("data/svd-256.pkl", "wb"))

    print("fitting logistic regressor")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    print("saving logistic regressor model")
    pickle.dump(lr, open("data/logreg.pkl", "wb"))
    print("finished fitting models")

df_test = pd.read_parquet("data/test.parquet")

X_test = df_test["tokens"]
y_test = df_test["class"]

X_test = svd.transform(tfidf.transform(X_test))
y_pred = lr.predict(X_test)

print(classification_report(y_test, y_pred))
