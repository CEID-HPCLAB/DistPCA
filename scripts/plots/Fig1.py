import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

workers = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64])

genomes_1000 =  np.array([356.7407945, 178.3993832, 92.95216167, 47.18473489, 
                          31.16004802, 23.48908567, 16.37145399, 12.43822264, 
                          8.997754257, 6.510368676])
genomes_50K =  np.array([110020.5023 * 1.9983, 110020.5023, 55154.07626, 27970.82618, 
                          18651.04972, 14180.5183, 9587.767641, 7084.838065, 
                          4913.973162, 3777.552357])
genomes_500K =  np.array([33297.95211 * 1.99987, 33297.95211, 16678.12446, 8412.132026, 
                           5637.877179, 4317.972034, 2949.662772, 2182.929662, 
                           1452.335722, 1097.604851])
genomes_1M =  np.array([37689.17871 * 1.9973, 37689.17871, 18836.35089, 9436.06865, 
                         6295.296339, 4814.640775, 3208.694778, 2396.712454, 
                         1610.854153, 1205.178123])

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

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

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
    
    ax.bar(ranks, times_plot, color=color, edgecolor='black', linewidth=1, width=0.7, zorder=3)
    
    ax.set_xlabel("MPI Ranks", fontsize = 11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    
    ax.set_xticks(ranks)
    ax.set_xticklabels(workers, fontsize=10)
    # ax.set_ylabel(y_label, fontsize=12, rotation=90, labelpad=10)
    
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

axes[0].set_ylabel("Wall-Clock Time", labelpad=11, fontsize=14)

legend_handles = [Patch(facecolor=color, edgecolor='black', label=name) 
                  for name, color in DATASET_COLORS.items()]

# fig.legend(handles=legend_handles, loc='upper center', ncol=4, frameon=True, shadow=True, edgecolor='black', facecolor='white',
#            bbox_to_anchor=(0.5, 1.11), framealpha=1.0, fontsize=12.5,  columnspacing = 1.5, borderpad=0.4)

fig.legend(handles=legend_handles, loc='upper center', ncol=4, frameon=True, shadow=True, edgecolor='white', facecolor='#1a1a1a',
           bbox_to_anchor=(0.5, 1.11), framealpha=1.0, fontsize=12.5,  columnspacing = 1.5, borderpad=0.4)

for ax in axes:
    ax.grid(True, axis='y', linestyle='--', linewidth=0.62, alpha=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        # spine.set_color('black')
        spine.set_color('white')

plt.tight_layout()
# plt.savefig("Fig1.pdf", dpi=600, bbox_inches="tight", transparent=True)
plt.savefig("Fig1.png", dpi=600, bbox_inches="tight", transparent=True)