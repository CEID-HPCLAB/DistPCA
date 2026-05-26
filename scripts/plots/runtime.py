import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

GENOMES_1000_PATH = "../../docs/results/runtime/1000_genomes.txt"
GENOMES_50K_PATH  = "../../docs/results/runtime/50K_genomes.txt"
GENOMES_500K_PATH = "../../docs/results/runtime/500K_genomes.txt"
GENOMES_1M_PATH   = "../../docs/results/runtime/1M_genomes.txt"

def load_dataset(path):
    workers, times = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            mpi = int(parts[0].split(":")[1].strip())
            t = float(parts[1].split(":")[1].strip().split()[0])

            workers.append(mpi)
            times.append(t)

    return np.array(workers), np.array(times)

workers, genomes_1000 = load_dataset(GENOMES_1000_PATH)
_, genomes_50K = load_dataset(GENOMES_50K_PATH)
_, genomes_500K = load_dataset(GENOMES_500K_PATH)
_, genomes_1M = load_dataset(GENOMES_1M_PATH)

datasets = {
    "1000 Genomes": genomes_1000,
    "50K Genomes": genomes_50K,
    "500K Genomes": genomes_500K,
    "1M Genomes": genomes_1M
}

DATASET_COLORS = {
    "1000 Genomes": "forestgreen",
    "50K Genomes":  "#4C72B0",
    "500K Genomes": "darkorange",
    "1M Genomes":   "darkred",
}

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

fig, axes = plt.subplots(1, len(datasets), figsize=(14, 3.15), sharey=False, dpi=600)

for ax, (name, times) in zip(axes, datasets.items()):
    color = DATASET_COLORS[name]
    ranks = np.arange(1, len(workers) + 1)

    if name == "1000 Genomes":
        times_plot = times
        y_label = "Time (s)"
        formatter = FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}s')
    else:
        times_plot = times / 3600
        formatter = FuncFormatter(lambda x, _: '' if x == 0 else (f'{int(x)}h' if x % 1 == 0 else f'{x:.1f}'))
    
    ax.bar(ranks, times_plot, color = color, edgecolor = 'black', linewidth = 1, width = 0.7, zorder = 3)
    
    ax.set_xlabel("MPI Ranks", fontsize = 11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    
    ax.set_xticks(ranks); ax.set_xticklabels(workers, fontsize = 10)
    
    if formatter is not None:
        ax.yaxis.set_major_formatter(formatter)

    if name == "500K Genomes":
        ax.set_ylim(0, 22)
        yticks = np.sort(np.concatenate(([2], np.arange(0, 21, 5))))
        ax.set_yticks(yticks)
    
    if name == "50K Genomes":
        ax.set_ylim(0, 66)
        yticks = np.sort(np.concatenate(([5], np.arange(0, 65, 10))))
        ax.set_yticks(yticks)
    
    if name == "1000 Genomes":
        ax.set_ylim(0, 440)
        ax.set_yticks(np.arange(0, 401, 100))
        yticks = np.sort(np.concatenate(([20, 60, 150,], np.arange(0, 401, 100))))
        ax.set_yticks(yticks)

    if name == "1M Genomes":
        ax.set_ylim(0, 22)
        yticks = np.sort(np.concatenate(([2], np.arange(0, 21, 5))))
        ax.set_yticks(yticks)

axes[0].set_ylabel("Wall-Clock Time", labelpad = 11, fontsize = 14)

legend_handles = [Patch(facecolor = color, edgecolor = 'black', label = name) for name, color in DATASET_COLORS.items()]

fig.legend(handles = legend_handles, loc = 'upper center', ncol = 4, frameon = True, shadow = True, 
           edgecolor = 'white', # black
           facecolor = '#1a1a1a', # white
           bbox_to_anchor = (0.5, 1.11), framealpha = 1.0, fontsize = 12.5,  columnspacing = 1.5, borderpad = 0.4)

for ax in axes:
    ax.grid(True, axis = 'y', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        # spine.set_color('black')
        spine.set_color('white')

plt.tight_layout()
# plt.savefig("runtime.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig("runtime.png", dpi = 600, bbox_inches = "tight", transparent = True)