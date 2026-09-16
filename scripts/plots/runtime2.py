import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

GENOMES_1000_PATH = "../../docs/results/runtime/athena/1000_genomes.txt"
GENOMES_500K_PATH = "../../docs/results/runtime/athena/500K_genomes.txt"

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
_, genomes_500K = load_dataset(GENOMES_500K_PATH)

datasets = {
    "1000 Genomes": genomes_1000,
    "500K Genomes": genomes_500K
}

DATASET_COLORS = {
    "1000 Genomes": "forestgreen",
    "500K Genomes": "darkorange",
}

HATCH_STYLE = "//"

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

fig, axes = plt.subplots(1, len(datasets), figsize = (6.7, 2.85), sharey = False, dpi = 600)

for ax, (name, times) in zip(axes, datasets.items()):
    color = DATASET_COLORS[name]
    ranks = np.arange(1, len(workers) + 1)

    if name == "1000 Genomes":
        times_plot = times
        formatter = FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}s')
    else:
        times_plot = times / 3600
        formatter = FuncFormatter(lambda x, _: f'{int(x)}h' if x.is_integer() else f'{x:.1f}h')

        ax.set_yticks(np.concatenate(([1], np.arange(2, 8, 2))))

    ax.bar(ranks, times_plot, color = color, edgecolor = 'black', hatch = HATCH_STYLE, linewidth = 1, width = 0.65, zorder = 3)

    ax.set_xlabel("MPI Ranks", fontsize = 11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)

    ax.set_xticks(ranks); ax.set_xticklabels(workers, fontsize = 10)
    ax.yaxis.set_major_formatter(formatter)

axes[0].set_ylabel("Wall-Clock Time", labelpad = 11, fontsize = 14)

legend_handles = [Patch(facecolor = color, edgecolor = 'black', hatch = HATCH_STYLE, label = name) for name, color in DATASET_COLORS.items()]

fig.legend(handles = legend_handles, loc = 'upper center', ncol = 2, frameon = True, shadow = True, 
           edgecolor = 'white', # black
           facecolor = '#1a1a1a', # white
           bbox_to_anchor = (0.5, 1.15), framealpha = 1.0, fontsize = 12,  columnspacing = 1.5, borderpad = 0.4)

for ax in axes:
    ax.grid(True, axis = 'y', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        # spine.set_color('black')
        spine.set_color('white')

plt.tight_layout()
# plt.savefig("runtime2.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig("runtime2.png", dpi = 600, bbox_inches = "tight", transparent = True)