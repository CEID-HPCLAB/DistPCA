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
colors = ["forestgreen",   "#4C72B0", "darkorange", "darkred"]

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 13

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

def linear_scale(values, ticks):
    positions = np.arange(len(ticks))
    scaled = []
    for v in values:
        for i in range(len(ticks)-1):
            if ticks[i] <= v <= ticks[i+1]:
                scaled_val = positions[i] + (v - ticks[i]) / (ticks[i+1] - ticks[i])
                scaled.append(scaled_val)
                break
        else:
            scaled.append(positions[-1])
    return np.array(scaled)

speedups = [d[0]/d for d in datasets]

fig, ax = plt.subplots(figsize = (8.5, 5.1), dpi = 600)

for sp, c, l in zip(speedups, colors, labels):
    ax.plot(np.arange(len(workers)), linear_scale(sp, workers), marker = 'o', color = c, label = l, linewidth = 2)

ax.plot(np.arange(len(workers)), np.arange(len(workers)), linestyle = '--', color = 'midnightblue', label = 'Ideal', linewidth = 2)

ax.set_xlabel('MPI Ranks', fontsize = 18, labelpad = 10); ax.set_ylabel('Speedup', fontsize = 18)
ax.set_xticks(np.arange(len(workers))); ax.set_xticklabels(workers, fontsize = 18)
ax.set_yticks(np.arange(len(workers))); ax.set_yticklabels(workers, fontsize = 18)

dataset_handles = [Line2D([0], [0], color=colors[i], marker='o', lw=2) for i in range(len(colors))]
dataset_legend = ax.legend(handles = dataset_handles, labels = labels,
                           loc = 'upper left', frameon = True, 
                           facecolor = '#1a1a1a', # white for light mode
                           edgecolor = 'white', # black for light mode
                           ncols = 2, framealpha = 1.0, shadow = True, fontsize = 12.5, labelspacing = 0.09, 
                           columnspacing = 1.02, bbox_to_anchor = (0.005, 1.003))

ax.add_artist(dataset_legend)

ideal_handle = [Line2D([0], [0], color = 'midnightblue', linestyle = '--', lw = 2)]
ax.legend(handles = ideal_handle, labels = ['Ideal Speedup'], loc = 'upper left', frameon = True, 
          facecolor = '#1a1a1a', # white for light mode
          edgecolor = 'white', # black for light mode
          framealpha = 1.0, shadow = True, fontsize = 13.5, bbox_to_anchor = (0.005, 0.867))

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    # spine.set_color('black') # black for light mode
    spine.set_color('white')

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
# plt.savefig("speedup.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig("speedup.png", dpi = 600, bbox_inches = "tight", transparent = True)