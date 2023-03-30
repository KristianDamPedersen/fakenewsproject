import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Function to extract second element of each sublist if required
def extract_second_element(y_probs):
    if isinstance(y_probs[0], list) or isinstance(y_probs[0], np.ndarray):
        y_probs = [prob[1] for prob in y_probs]
    return y_probs


# Load predictions and true values
simple_y_probs = np.load("data/predictions/simple_y_probs.npy")
simple_y_true = np.load("data/predictions/simple_y_true.npy")
simple_y_probs = extract_second_element(simple_y_probs)

big_dnn_y_preds = np.load("data/predictions/big_dnn_y_preds.npy")
big_dnn_y_true = np.load("data/predictions/big_dnn_y_true.npy")
big_dnn_y_preds = extract_second_element(big_dnn_y_preds)

dnn_y_preds = np.load("data/predictions/dnn_y_preds.npy")
dnn_y_true = np.load("data/predictions/dnn_y_true.npy")
# dnn_y_preds = extract_second_element(dnn_y_preds)

xgboost_y_probs = np.load("data/predictions/xgboost_y_probs.npy")
xgboost_y_true = np.load("data/predictions/xgboost_y_true.npy")
xgboost_y_probs = extract_second_element(xgboost_y_probs)

# Define the models
models = [
    {
        "name": "Simple Model",
        "probs": simple_y_probs,
        "true": simple_y_true,
        "color": "darkorange",
    },
    {
        "name": "Big DNN",
        "probs": big_dnn_y_preds,
        "true": big_dnn_y_true,
        "color": "forestgreen",
    },
    {
        "name": "Small DNN",
        "probs": dnn_y_preds,
        "true": dnn_y_true,
        "color": "darkmagenta",
    },
    {
        "name": "XGBoost",
        "probs": xgboost_y_probs,
        "true": xgboost_y_true,
        "color": "indianred",
    },
]

# Plot the ROC curve
plt.figure()

# Plot ROC curves for each model
for model in models:
    fpr, tpr, _ = roc_curve(model["true"], model["probs"])
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color=model["color"],
        lw=1,
        label=f"{model['name']} (AUC = {roc_auc:.3f})",
    )


plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC on FakeNews dataset")
plt.legend(loc="lower right")
plt.savefig("report/src/figures/ROC_curves_fakenews.png")
