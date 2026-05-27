import pandas as pd
import matplotlib.pyplot as plt
 
PC_FILE_PATH = "../../docs/results/accuracy/1000_genomes_leftSingularVectors.txt"
PANEL_FILE_PATH = "../../docs/results/accuracy/integrated_call_samples_v3.20130502.ALL.panel"

df = pd.read_csv(PC_FILE_PATH, sep = r"\s+", engine = "python")

panel = pd.read_csv(PANEL_FILE_PATH, sep = "\t")[["sample", "super_pop"]]
panel.columns = ["FID", "continent"]

df = df.merge(panel, on = "FID", how = "left").dropna(subset = ["continent"])

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.labelsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13
})

plt.rcParams['text.color'] = 'white'; plt.rcParams['axes.labelcolor'] = 'white'; plt.rcParams['xtick.color'] = 'white'; plt.rcParams['ytick.color'] = 'white'

# plt.rcParams['text.color'] = 'black'; plt.rcParams['axes.labelcolor'] = 'black'; plt.rcParams['xtick.color'] = 'black'; plt.rcParams['ytick.color'] = 'black'

fig, ax = plt.subplots(figsize = (8.5, 5.1), dpi = 600)

palette = {
    "AFR": "#1f77b4",
    "EUR": "#d62728",
    "EAS": "#2ca02c",
    "SAS": "#ff7f0e",
    "AMR": "#9467bd"
}

continents = sorted(df["continent"].unique())

for c in continents:
    subset = df[df["continent"] == c]
    ax.scatter(subset["PC0"], subset["PC1"],
        label = c, color = palette.get(c, "gray"),
        alpha = 0.6, edgecolors = "white",
        linewidths = 0.3, s = 50
    )

ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines['left'].set_linewidth(1.2); ax.spines['bottom'].set_linewidth(1.2)

# ax.spines['left'].set_color('black')
# ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')

ax.grid(True, axis = 'both', linestyle = '--', linewidth = 0.62, alpha = 0.55, zorder = 0)

ax.set_yticks([-0.02, 0.00, 0.02]); ax.set_yticklabels(["-0.02", "0.00", "0.02"])

ax.legend(loc = "lower right", frameon = True,
    facecolor = '#1a1a1a', # white for light mode
    edgecolor = 'white', # black for light mode
    framealpha = 1.0, shadow = True,
    ncol = len(continents), markerscale = 1.5, labelspacing = 0.09,
    columnspacing = 1.02, fontsize = 13, bbox_to_anchor = (0.995, 0.01)
)

plt.tight_layout()
# plt.savefig("pop_structure.pdf", dpi = 600, bbox_inches = "tight", transparent = True)
plt.savefig("pop_structure.png", dpi = 600, bbox_inches = "tight", transparent = True)