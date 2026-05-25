import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

workers = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64])

genomes_1000 =  np.array([356.7407945, 178.3993832, 92.95216167, 47.18473489, 31.16004802, 23.48908567, 16.37145399, 12.43822264, 8.997754257, 6.510368676])
genomes_50K =  np.array([110020.5023 * 1.9983, 110020.5023, 55154.07626, 27970.82618, 18651.04972, 14180.5183, 9587.767641, 7084.838065, 4913.973162, 3777.552357])
genomes_500K    =  np.array([33297.95211 * 1.99987, 33297.95211, 16678.12446, 8412.132026, 5637.877179, 4317.972034, 2949.662772, 2182.929662, 1452.335722, 1097.604851])
genomes_1M     =  np.array([37689.17871 * 1.9973, 37689.17871, 18836.35089, 9436.06865, 6295.296339, 4814.640775, 3208.694778, 2396.712454, 1610.854153, 1205.178123])

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