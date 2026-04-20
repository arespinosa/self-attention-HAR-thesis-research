import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")
X = np.load("Children_X.npy")                     # raw sensor data
X_pred = np.load("children_predictions.npy")      # model outputs (logits/probs)
print(f"X shape: {X.shape}")
print(f"X_pred shape: {X_pred.shape}")


# -----------------------------
# PREDICTIONS (PAMAP2 SPACE)
# -----------------------------
print("Computing predicted classes...")
pred_classes = np.argmax(X_pred, axis=1)


# -----------------------------
# PAMAP2 → CHILDREN MAPPING
# -----------------------------
pamap2_to_children = {
    1: 5,   # lying
    2: 3,   # sitting
    3: 4,   # standing
    4: 0,   # walking
    5: 1,   # running
    6: 2,   # cycling
    7: 6,   # other
    11: 6,
    12: 6,
    13: 6,
    14: 6,
    18: 6
}

children_map = {
    0: "walking",
    1: "running",
    2: "cycling",
    3: "sitting",
    4: "standing",
    5: "lying",
    6: "other"
}

print("Mapping predictions to children label space...")
mapped_preds = np.array([
    pamap2_to_children.get(p, 6)
    for p in pred_classes
])


# -----------------------------
# UMAP FUNCTION
# -----------------------------
def reduceme(X, name=""):
    print(f"\n[UMAP] Processing {name}...")

    if len(X.shape) == 3:
        print("Flattening time-series windows...")
        X = X.reshape(X.shape[0], -1)

    print("Scaling data...")
    X = StandardScaler().fit_transform(X)

    print("Fitting UMAP (this may take a while)...")
    reducer = umap.UMAP(n_components=2, random_state=42)

    X_2d = reducer.fit_transform(X)

    print(f"[UMAP] Done with {name}")
    return X_2d


# -----------------------------
# PLOT FUNCTION
# -----------------------------
def scatter_plot(X_2d, labels, title):
    print(f"Plotting: {title}")

    plt.figure(figsize=(7, 6))
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")

    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=10,
            color=cmap(i % 20),
            label=children_map.get(label, str(label)),
            alpha=0.8
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    print(f"Finished plot: {title}")


# -----------------------------
# 1. RAW DATA UMAP (colored by predictions)
# -----------------------------
X_raw_umap = reduceme(X, name="Raw Sensor Data")

scatter_plot(
    X_raw_umap,
    mapped_preds,
    "Children Raw Data (Colored by Predictions)"
)


# -----------------------------
# 2. PREDICTION SPACE UMAP (optional)
# -----------------------------
# This shows clustering in prediction space (less important but still useful)
X_pred_umap = reduceme(X_pred, name="Prediction Vectors")

scatter_plot(
    X_pred_umap,
    mapped_preds,
    "Prediction Space (UMAP)"
)


print("\nAll done!")