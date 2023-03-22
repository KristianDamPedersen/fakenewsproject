import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from custom_tokeniser import custom_tokenizer
import pickle

def pass_fun(doc):
    return doc


print("Loading models")

tfidf = pickle.load(open("tfidf.pkl", "rb"))
svd = pickle.load(open("svd.pkl", "rb"))
lr = pickle.load(open("lr.pkl", "rb"))

print("Finished loading models")

df_test = pd.read_parquet("test.parquet")

X_test = df_test["tokens"]
y_test = df_test["class"]

X_test = svd.transform(tfidf.transform(X_test))
score = lr.score(X_test, y_test)
print(score)
