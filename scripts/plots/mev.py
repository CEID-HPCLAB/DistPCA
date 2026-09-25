import os
import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DRIVE_URL = "https://drive.google.com/drive/folders/1FATyewsgDSeSHo10kqfXkh2qxlurDmEo?usp=sharing"
BASE_PATH = "../../docs/results/accuracy/mev"

def download_res():
    if os.path.exists(BASE_PATH) and os.listdir(BASE_PATH):
        print(f"MEV results already available in: {BASE_PATH}")
        return
        
    print("Downloading MEV results from Google Drive...")
    gdown.download_folder(url = DRIVE_URL, output = BASE_PATH, quiet = False, use_cookies = False,)

def load_pcaone(path, k):
    df = pd.read_csv(path, sep = r"\s+")
    df["IID"] = df["IID"].astype(str).str.strip()

    pc_cols = [f"PC{i}" for i in range(1, k + 1)]
    df = df.set_index("IID")[pc_cols]; df.columns = [f"PC{i}" for i in range(k)]

    return df

def load_distpca(path, k):
    df = pd.read_csv(path, sep = r"\s+")
    id_col = df.columns[0]

    df[id_col] = df[id_col].astype(str).str.strip()
    pc_cols = [f"PC{i}" for i in range(k)]

    df = df.set_index(id_col)[pc_cols]; df.index.name = "IID"

    return df

def compute_mev(dataset, k):
    pcaone_path = f"{BASE_PATH}/PCAone/{dataset['folder']}/{k}.eigvecs2"
    distpca_path = f"{BASE_PATH}/DistPCA/{dataset['folder']}/{k}_leftSingularVectors.txt"

    pcaone_PCs = load_pcaone(pcaone_path, k); distpca_PCs = load_distpca(distpca_path, k)
    common = pcaone_PCs.index.intersection(distpca_PCs.index)

    pcaone_PCs = pcaone_PCs.loc[common].sort_index()
    distpca_PCs = distpca_PCs.loc[common].sort_index()

    return np.linalg.norm(pcaone_PCs.values.T @ distpca_PCs.values, ord = "fro") ** 2 / pcaone_PCs.values.shape[1]

K_VALS = [2, 5, 10, 15, 20]
DATASETS = [{"folder": "500K-Genomes", "label": "500K Genomes", "color": "darkorange", "marker": "D"},
            {"folder": "1M-Genomes", "label": "1M Genomes", "color": "darkred", "marker": "^"}]

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

download_res()

results = []

for dataset in DATASETS:

    mev_vals = []

    for k in K_VALS:

        mev_vals.append(compute_mev(dataset, k))

    results.append((dataset["label"], dataset["color"], dataset["marker"], mev_vals))

fig, ax = plt.subplots(figsize = (8.5, 4.5), dpi = 600)

x = np.arange(1, len(K_VALS) + 1)

for label, color, marker, mev_values in results:
    ax.plot(x, mev_values, linestyle = "--", marker = marker, markersize = 9, linewidth = 2.8, color = color, label = label)

ax.set_xticks(x); ax.set_xticklabels([str(k) for k in K_VALS])
ax.set_xlabel(r"Number of Estimated PCs ($k$)"); ax.set_ylabel("Accuracy (MEV)")

ax.set_ylim(0.9945, 1.00035); ax.set_yticks(np.arange(0.995, 1.001, 0.001))

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)

ax.legend(loc = "lower right", frameon = True,
    edgecolor = 'white', # black
    facecolor = '#1a1a1a', # white
    framealpha = 1.0, shadow = True, fontsize = 13.5, bbox_to_anchor = (0.96, 0.018)
)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    # spine.set_color('black') # black for light mode
    spine.set_color('white')

ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.tight_layout()
# plt.savefig("mev.pdf", dpi = 600, bbox_inches = "tight")
plt.savefig("mev.png", dpi = 600, bbox_inches = "tight", transparent = True)