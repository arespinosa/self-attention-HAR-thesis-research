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
    return X_red, reducer, scaler


def scatter_plot(X, Y, title, single_point=None, single_label=None):
    plt.figure(figsize=(7, 6))
    labels_unique = np.unique(Y)

    cmap = plt.get_cmap("tab20")  # 20 distinct colors
    color_map = {label: cmap(i % 20) for i, label in enumerate(labels_unique)}

    for i, label in enumerate(labels_unique):
        idx = Y == label
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            s=10,
            color=color_map[label],
            label=str(label),
            alpha=0.8
        )

    if single_point is not None:
        plt.scatter(
        single_point[0, 0],
        single_point[0, 1],
        s=300,
        marker='*',
        color='black',
        edgecolors='white',
        linewidths=1.5,
        label=single_label
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
single_embedding = np.load("single_embedding.npy")

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
labels_combined = np.array([f"{i} - {name}" for i, name in zip(labels_int, labels_names)])
X_raw_red, _, _ = reduceme(raw_X)
scatter_plot(X_raw_red, labels_combined, "Raw Input (UMAP)")

X_ssl_red, reducer, scaler = reduceme(ssl_feats)

single_scaled = scaler.transform(single_embedding.reshape(1, -1))
single_red = reducer.transform(single_scaled)

scatter_plot(
    X_ssl_red,
    labels_combined,
    "SSL Pretrained Features (UMAP)",
    single_point=single_red,
    single_label="Random Sample - 6 - cycling"
)
