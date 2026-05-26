import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

datasets = [genomes_1000, genomes_50K, genomes_500K, genomes_1M]
labels = ["1000 Genomes", "50K Genomes", "500K Genomes", "1M Genomes"]
colors = ["forestgreen", "#4C72B0", "darkorange", "darkred"]

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

efficiencies = [(d[0] / d) / workers for d in datasets]

fig, ax = plt.subplots(figsize = (8.5, 5.1), dpi = 600)

for eff, c, l in zip(efficiencies, colors, labels):
    ax.plot(np.arange(len(workers)), eff, marker = 'o', color = c, linewidth = 2, label = l)

ax.plot(np.arange(len(workers)), np.ones_like(workers), linestyle = '--', color = 'midnightblue', linewidth = 2)

ax.set_xlabel('MPI Ranks', fontsize = 18, labelpad = 10)
ax.set_ylabel('Parallel Efficiency', fontsize = 18)

ax.set_xticks(np.arange(len(workers)))
ax.set_xticklabels(workers, fontsize = 18)

ax.set_yticks(np.linspace(0, 1, 6))
ax.set_yticklabels([f"{x*100:.0f}%" for x in np.linspace(0, 1, 6)], fontsize = 18)

ax.set_ylim(0, 1.05)

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('white') 
    # spine.set_color('black')

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

dataset_handles = [Line2D([0], [0], color = colors[i], marker = 'o', lw = 2) for i in range(len(colors))]

dataset_legend = ax.legend(handles = dataset_handles,
    labels = labels, loc = 'lower left', frameon = True,
    facecolor = '#1a1a1a', # white
    edgecolor = 'white', # black
    ncols = 2, framealpha = 1.0, shadow = True,
    fontsize = 12.5, bbox_to_anchor = (0.005, 0.02)
)

ax.add_artist(dataset_legend)

ideal_handle = [Line2D([0], [0], color = 'midnightblue', linestyle = '--', lw = 2)]

ax.legend(handles = ideal_handle, labels = ['Ideal Efficiency'],
    loc = 'lower left', frameon = True,
    facecolor = '#1a1a1a', # white
    edgecolor = 'white', # black
    framealpha = 1.0, shadow = True, fontsize = 13.5,
    bbox_to_anchor = (0.005, 0.18)
)

plt.tight_layout()
# plt.savefig("efficiency.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig("par_efficiency.png", dpi = 600, bbox_inches = "tight", transparent = True)