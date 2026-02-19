import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


def reduceme(X, method='umap'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))

    if method == 'umap':
        print("Fitting UMAP...")
        # Match repo exactly — no explicit hyperparameters
        reducer = umap.UMAP()
    else:
        print("Fitting PCA...")
        reducer = PCA(n_components=2)

    X_red = reducer.fit_transform(X_scaled)
    return X_red


def scatter_plot(X, Y, title, combined_labels=False):
    plt.figure(figsize=(7, 6))

    if not combined_labels:
        labels_unique = np.unique(Y)
    else:
        labels_unique = np.unique(Y)

    for label in labels_unique:
        idx = Y == label
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            s=10,
            label=str(label)
        )

    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    # Move legend outside
    plt.legend(
        bbox_to_anchor=(1.05, 1),  # right of the plot
        loc='upper left',
        borderaxespad=0.,
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(f"figs/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot: figs/{title.replace(' ', '_')}.png")
    plt.show()



raw_X = np.load("pamap2_raw.npy")           
ssl_feats = np.load("pamap2_embeddings.npy")
labels = np.load("pamap2_labels.npy")

labels_int = np.argmax(labels, axis=1)

# Map to activity names
col_to_activity = {
    0: "other",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    8: "other",
    9: "other",
    10: "other",
    11: "ascending stairs",
    12: "descending stairs",
    13: "vacuum cleaning",
    14: "ironing",
    15: "other",
    16: "other",
    17: "other",
    18: "rope jumping"
}

labels_names = np.array([col_to_activity[i] for i in labels_int])

X_raw_red = reduceme(raw_X)
X_ssl_red = reduceme(ssl_feats)

labels_combined = np.array([f"{i} - {name}" for i, name in zip(labels_int, labels_names)])

scatter_plot(X_raw_red, labels_combined, "Raw Input (UMAP)")
scatter_plot(X_ssl_red, labels_combined, "SSL Pretrained Features (UMAP)")

