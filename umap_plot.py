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

tfidf = pickle.load(open("tfidf-2048.pkl", "rb"))
svd = pickle.load(open("svd.pkl", "rb"))

print("loading df")
# Load the data
df = pd.read_parquet("small_train.parquet", columns=["tokens", "type"])
print("finished loading")

# Try to grab 5000 of each type
df = df.groupby("type").head(12000)

liar_df = pd.read_parquet("liar/liar.parquet", columns=["tokens", "type"])

X_train = svd.transform(tfidf.transform(df["tokens"]))
# y_train are the labels, "politics", "sports", etc.
# Fit umap
print("fitting umap")
try:
    reducer = pickle.load(open("umap.pkl", "rb"))
except:
    reducer = umap.UMAP(random_state=42, n_neighbors=25, min_dist=0.1, n_components=2)
    reducer.fit_transform(X_train)
    pickle.dump(reducer, open("umap.pkl", "wb"))

df = pd.concat([df, liar_df])
classes = df["type"].unique()
df = df.sample(frac=1)
y_train = df["type"]

embedding = reducer.transform(svd.transform(tfidf.transform(df["tokens"])))

print("done fitting umap")
# Replace each label in y_train with a number

# Assign each class a number
target = np.zeros(y_train.shape, dtype=int)
for i, c in enumerate(classes):
    target[y_train == c] = i

colours = [
    "yellow",  # Rumor
    "black",  # Political
    "blue",  # Junk science
    "green",  # RELIABLE
    "red",  # Fake
    "orange",  # Biased
    "#666040",  # Unrealiable
    "#6600ff",  # Hate
    "cyan",  # Conspiracy
    "grey",  # Clickbait
    "purple",  # Satirical
    "#ff00ff",
]  # Liar Data (All classes)
# Plot
plt.scatter(
    *embedding.T, c=[colours[i] for i in target], s=0.05, alpha=0.25, marker=",", lw=0
)

plt.legend(
    handles=[
        mpatches.Patch(color=colours[i], label=classes[i]) for i in range(len(classes))
    ]
)

plt.savefig("liar.png", dpi=3600)
