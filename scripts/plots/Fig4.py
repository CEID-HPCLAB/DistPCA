import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pc_file = "../../docs/results/noprefix_singularVectors.txt"
panel_file = "../../docs/results/integrated_call_samples_v3.20130502.ALL.panel"

df = pd.read_csv(pc_file, sep=r"\s+", engine="python")
panel = pd.read_csv(panel_file, sep="\t")[["sample", "pop"]]
panel.columns = ["FID", "population"]
df = df.merge(panel, on="FID", how="left").dropna(subset=["population"])

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.labelsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13
})

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

fig, ax = plt.subplots(figsize=(8.5, 5.1), dpi=600)

pops = sorted(df["population"].unique())
colors = sns.color_palette("Spectral", len(pops))

for i, p in enumerate(pops):
    subset = df[df["population"] == p]
    ax.scatter(
        subset["PC0"], 
        subset["PC1"], 
        label=p, 
        color=colors[i], 
        alpha=0.6, 
        edgecolors="white", 
        linewidths=0.3, s=50
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
# ax.spines['left'].set_color('black')
# ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.grid(True, axis='both', linestyle='--', linewidth=0.62, alpha=0.55, zorder=0)

# ax.legend(
#     loc="lower right",
#     frameon=True,
#     facecolor="white",
#     edgecolor="black",
#     framealpha=1.0,
#     shadow=True,
#     ncol=4,
#     markerscale=1.5,
#     labelspacing=0.09,
#     columnspacing=1.02,
#     fontsize=12.5
# )
ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="#1a1a1a",
    edgecolor="white",
    framealpha=1.0,
    shadow=True,
    ncol=4,
    markerscale=1.5,
    labelspacing=0.09,
    columnspacing=1.02,
    fontsize=12.5
)

plt.tight_layout()
# plt.savefig("Fig4.pdf", dpi=600, bbox_inches="tight", transparent=True)
plt.savefig("Fig4.png", dpi=600, bbox_inches="tight", transparent=True)
plt.show()