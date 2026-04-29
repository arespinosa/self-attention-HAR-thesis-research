import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# FILE LIST
# -----------------------------
base_names = [
    "P009_QM-SS1_0E3E9_p1",
    "P009_QM-SS1_0E3E9_p2",
    "P009_QM-SS1_11CCD_p1",
    "P009_QM-SS1_11CCD_p2",
    "P009_QM-SS1_14A51_p1",
    "P009_QM-SS1_14A51_p2",
    "P009_QM-SS1_16E17_p1",
    "P009_QM-SS1_16E17_p2",
    "P009_QM-SS1_1503C_p1",
    "P009_QM-SS1_1503C_p2",
    "P009_QM-SS1_12144_p1",
    "P009_QM-SS1_12144_p2"
]

# -----------------------------
# LABEL MAPPINGS
# -----------------------------
pamap2_to_children = {
    1: 5,   # lying
    2: 3,   # sitting
    3: 4,   # standing
    4: 0,   # walking
    5: 1,   # running
    6: 6,   # cycling → other
    7: 6,   # Nordic walking → other
    11: 6,
    12: 6,
    13: 6,
    14: 6,
    18: 2   # rope jumping
}

children_map = {
    0: "walking",
    1: "running",
    2: "rope jumping",
    3: "sitting",
    4: "standing",
    5: "lying",
    6: "other"
}

# -----------------------------
# FIXED COLOR MAP (CONSISTENT)
# -----------------------------
activity_colors = {
    0: "#1f77b4",  # walking - blue
    1: "#d62728",  # running - red
    2: "#2ca02c",  # rope jumping - green
    3: "#ff7f0e",  # sitting - orange
    4: "#9467bd",  # standing - purple
    5: "#8c564b",  # lying - brown
    6: "#7f7f7f"   # other - gray
}

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

    print("Fitting UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    return X_2d


# -----------------------------
# PLOT FUNCTION (CONSISTENT COLORS)
# -----------------------------
def scatter_plot(X_2d, labels, title, save_name):
    print(f"Plotting: {title}")

    plt.figure(figsize=(7, 6))

    # Loop over ALL possible labels (ensures consistency)
    for label in children_map.keys():
        idx = labels == label

        if np.sum(idx) == 0:
            # Invisible point just for legend consistency
            plt.scatter([], [],
                        color=activity_colors[label],
                        label=children_map[label])
        else:
            plt.scatter(
                X_2d[idx, 0],
                X_2d[idx, 1],
                s=10,
                color=activity_colors[label],
                label=children_map[label],
                alpha=0.8
            )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title="Activities"
    )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    # Ensure directory exists
    os.makedirs("figs_children", exist_ok=True)

    save_path = f"figs_children/{save_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Saved plot: {save_path}")
    plt.show()


# -----------------------------
# MAIN LOOP
# -----------------------------
for base in base_names:
    print(f"\n==============================")
    print(f"Processing: {base}")
    print(f"==============================")

    # Load data
    X = np.load(f"{base}_X.npy")
    X_pred = np.load(f"{base}_predictions.npy")

    print(f"X shape: {X.shape}")
    print(f"X_pred shape: {X_pred.shape}")

    # Predictions
    pred_classes = np.argmax(X_pred, axis=1)

    # Debug (optional but useful)
    print("Unique PAMAP2 preds:", np.unique(pred_classes))

    mapped_preds = np.array([
        pamap2_to_children.get(p, 6)
        for p in pred_classes
    ])

    print("Unique mapped labels:", np.unique(mapped_preds))

    # -----------------------------
    # RAW DATA UMAP
    # -----------------------------
    X_raw_umap = reduceme(X, name=f"{base} Raw")

    scatter_plot(
        X_raw_umap,
        mapped_preds,
        f"{base} Raw Data",
        save_name=f"{base}_raw"
    )

    # -----------------------------
    # PREDICTION SPACE UMAP
    # -----------------------------
    X_pred_umap = reduceme(X_pred, name=f"{base} Predictions")

    scatter_plot(
        X_pred_umap,
        mapped_preds,
        f"{base} Prediction Space",
        save_name=f"{base}_predictions"
    )

print("\nAll files processed!")