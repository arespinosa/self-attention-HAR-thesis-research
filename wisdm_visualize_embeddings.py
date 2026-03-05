import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


def reduceme(X, method='umap'):
    scaler = StandardScaler()

    # Flatten raw windows if needed
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], -1)

    X_scaled = scaler.fit_transform(X)

    if method == 'umap':
        print("Fitting UMAP...")
        reducer = umap.UMAP()  # match PAMAP2 exactly
    else:
        print("Fitting PCA...")
        reducer = PCA(n_components=2)

    X_red = reducer.fit_transform(X_scaled)
    return X_red, reducer, scaler


def scatter_plot(X, Y, title, single_point=None, single_label=None):
    plt.figure(figsize=(7, 6))
    labels_unique = np.unique(Y)

    cmap = plt.get_cmap("tab20")  # identical to PAMAP2
    color_map = {label: cmap(i % 20) for i, label in enumerate(labels_unique)}

    for label in labels_unique:
        idx = Y == label
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            s=10,
            color=color_map[label],
            label=str(label),
            alpha=0.8
        )

    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    # Legend outside plot (identical behavior)
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=8
    )

    # Ensure figs directory exists
    os.makedirs("figs_wisdm", exist_ok=True)

    save_path = f"figs_wisdm/{title.replace(' ', '_')}.png"
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.show()

raw_X = np.load("wisdm_experiments/wisdm_data/wisdm_30hz_clean/X.npy")
ssl_feats = np.load("wisdm_experiments/wisdm_data/wisdm_30hz_clean/SSL_feats.npy")
labels_raw = np.load("wisdm_experiments/wisdm_data/wisdm_30hz_clean/Y.npy")

print("Raw:", raw_X.shape)
print("SSL:", ssl_feats.shape)


activity_map = json.load(
    open(os.path.join("configs", "activity_maps", "wisdm.json"))
)

activity_map = {int(k): v for k, v in activity_map.items()}

if isinstance(labels_raw[0], str):
    reverse_map = {v: k for k, v in activity_map.items()}
    labels_int = np.array([reverse_map[label] for label in labels_raw])
else:
    labels_int = labels_raw

labels_names = np.array([activity_map[i] for i in labels_int])
labels_combined = np.array([
    f"{i} - {name}" for i, name in zip(labels_int, labels_names)
])

X_raw_red, _, _ = reduceme(raw_X, method='umap')
scatter_plot(X_raw_red, labels_combined, "WISDM Raw Input (UMAP)")

X_ssl_red, reducer, scaler = reduceme(ssl_feats, method='umap')
scatter_plot(X_ssl_red, labels_combined, "WISDM SSL Features (UMAP)")