# Same as other UMAP but the original dataset is greyed out and the colours are the various classes from the LIAR dataset.
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from custom_tokeniser import custom_tokenizer
from lib.pass_fun import pass_fun


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
classes = [
    "news",
    "true",
    "mostly-true",
    "half-true",
    "barely-true",
    "false",
    "pants-fire",
]
y_train = df["type"]


# We generate the word embedding
embedding = reducer.transform(svd.transform(tfidf.transform(df["tokens"])))

print("done transforming umap")
# Assign each class a number
target = np.zeros(y_train.shape, dtype=int)
for i, c in enumerate(classes):
    target[y_train == c] = i

colours = [
    "grey",  # Fake news corpus
    "#007700",  # True
    "#7FFF00",  # Mostly true
    "#DDDD00",  # Half true
    "#FF2400",  # Barely true
    "#FF003E",  # False
    "#0000FF",  # Pants on fire
]  # Liar Data (All classes)
# Scatter plot
plt.scatter(
    *embedding.T, c=[colours[i] for i in target], s=0.05, alpha=0.5, marker=",", lw=0
)

# Legends for the various classes
plt.legend(
    handles=[
        mpatches.Patch(color=colours[i], label=classes[i]) for i in range(len(classes))
    ]
)

# We save the figure, this image exists as an altered figure in the document
plt.savefig("liar_binary.png", dpi=3600)
