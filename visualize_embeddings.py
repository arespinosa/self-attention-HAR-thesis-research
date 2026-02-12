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
        reducer = umap.UMAP(n_components=2, random_state=42, verbose=True)
    else:
        print("Fitting UMAP...")
        reducer = PCA(n_components=2)

    X_red = reducer.fit_transform(X_scaled)
    return X_red


def scatter_plot(X, Y, title):
    plt.figure(figsize=(7, 6))

    unique_labels = np.unique(Y)

    for label in unique_labels:
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
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"figs/{title.replace(' ', '_')}.png", dpi=300)
    print(f"Saved plot: figs/{title.replace(' ', '_')}.png")
    plt.show()


raw_X = np.load("pamap2_raw.npy")           
ssl_feats = np.load("pamap2_embeddings.npy")
labels = np.load("pamap2_labels.npy")

labels_int = np.argmax(labels, axis=1)



X_raw_red = reduceme(raw_X)
X_ssl_red = reduceme(ssl_feats)


scatter_plot(X_raw_red, labels_int, "Raw Input (UMAP)")
scatter_plot(X_ssl_red, labels_int, "SSL Pretrained Features (UMAP)")
