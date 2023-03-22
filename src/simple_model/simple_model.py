import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from custom_tokeniser import custom_tokenizer
import pickle

df = pd.read_parquet("small_train.parquet", columns=["tokens", "class"])
X_train = df["tokens"]
y_train = df["class"]

print(y_train.value_counts())

def pass_fun(doc):
    return doc

tfidf = TfidfVectorizer(max_features=2048, sublinear_tf=True, preprocessor=pass_fun, tokenizer=pass_fun)

X_train = tfidf.fit_transform(X_train)
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

svd = TruncatedSVD(n_components=256, random_state = 42)
X_train = svd.fit_transform(X_train)
pickle.dump(svd, open("svd.pkl", "wb"))

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

pickle.dump(lr, open("lr.pkl", "wb"))

df_test = pd.read_parquet("test.parquet")

X_test = df_test["tokens"]
y_train = df_test["class"]

X_test = svd.transform(tfidf.transform(X_test))
score = lr.score(X_test, y_test)
print(score)
