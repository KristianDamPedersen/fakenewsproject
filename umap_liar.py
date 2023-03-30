# Same as umaptest but liar is not added in the UMAP fit

import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from custom_tokeniser import custom_tokenizer


def pass_fun(doc):
    return doc


print("loading models")

tfidf = pickle.load(open("data/tfidf-2048.pkl", "rb"))
svd = pickle.load(open("data/svd-256.pkl", "rb"))

print("loading df")
# Load the data
df = pd.read_parquet("data/small_train.parquet", columns=["tokens", "type"])
print("finished loading")

# Try to grab 5000 of each type
df = df.groupby("type").head(12000)

liar_df = pd.read_parquet("liar/liar.parquet", columns=["tokens", "label"])

liar_df["type"] = liar_df["label"]

X_train = svd.transform(tfidf.transform(df["tokens"]))
# y_train are the labels, "politics", "sports", etc.
# Fit umap
try:
    reducer = pickle.load(open("data/umap.pkl", "rb"))
except:
    print("fitting umap")
    reducer = umap.UMAP(random_state=420, n_neighbors=25, min_dist=0.1, n_components=2)
    reducer.fit_transform(X_train)
    pickle.dump(reducer, open("data/umap.pkl", "wb"))
    print("done fitting umap")
df["type"] = "news"

df = pd.concat([df, liar_df])
classes = ["news", "true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
y_train = df["type"]

embedding = reducer.transform(svd.transform(tfidf.transform(df["tokens"])))

print("done transforming umap")
# Replace each label in y_train with a number

# Assign each class a number
target = np.zeros(y_train.shape, dtype=int)
for i, c in enumerate(classes):
    target[y_train == c] = i

colours = [
    "grey",  # Fake news corpus
    "#005500",   # True
    "#229922",   # Mostly true
    "#669900",   # Half true
    "#884400",   # Barely true
    "#DD0000",   # False
    "#0000FF",   # Pants on fire
]  # Liar Data (All classes)
# Plot
plt.scatter(
    *embedding.T, c=[colours[i] for i in target], s=0.1, alpha=0.5, marker=",", lw=0
)

plt.legend(
    handles=[
        mpatches.Patch(color=colours[i], label=classes[i]) for i in range(len(classes))
    ]
)

plt.savefig("liar_binary.png", dpi=3600)
