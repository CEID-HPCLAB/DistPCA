import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

approx_file = "../../docs/results/noprefix_singularVectors.txt"
true_file   = "../../docs/results/noprefix_realLeftsingularVectors.txt"

approx_df = pd.read_csv(approx_file, sep=r"\s+", engine="python")
U_approx = approx_df.drop(columns=["FID"]).values
U_true = np.loadtxt(true_file)

m, k = U_approx.shape
eps = 1e-12

for i in range(k):
    if np.dot(U_approx[:, i], U_true[:, i]) < 0:
        U_approx[:, i] *= -1

rel_err = np.abs(U_approx - U_true) / (np.abs(U_true) + eps)

num_points = 10
idx = np.linspace(0, m - 1, num_points, dtype=int)

# plt.rcParams.update({"font.family": "serif", "font.size": 16})
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

markers = ["o", "x", "s", "D", "^", "v", ">", "<", "*", "P"]
colors = sns.color_palette("Paired", k)

fig, ax = plt.subplots(figsize=(8.5, 5.1), dpi=600)

for i in range(k):
    y_vals = rel_err[idx, i].copy()
    all_zero = y_vals.max() == 0

    y_vals = np.where(y_vals == 0, eps, y_vals)

    ax.plot(idx, y_vals,
            linestyle="--", marker=markers[i], linewidth=2,
            color=colors[i], label=f"PC{i+1}")

    # if all_zero:
    #     vertical_offset = 40 if i == 0 else 70
    #     horizontal_offset = 44 if i == 0 else 5
    #     ax.annotate(f"PC{i+1}: error ≈ 0",
    #                 xy=(idx[0], eps),
    #                 xytext=(horizontal_offset, vertical_offset),
    #                 textcoords="offset points",
    #                 fontsize=14, color=colors[i],
    #                 ha="left",
    #                 arrowprops=dict(arrowstyle="-", color=colors[i],
    #                                lw=1.2, linestyle="dotted"))

ax.set_yscale("log")

# Force fewer y-axis ticks
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))

# Adjust limits to avoid excessive whitespace if errors are much larger than eps
valid_data = rel_err[idx, :].flatten()
min_nonzero = valid_data[valid_data > 0].min() if any(valid_data > 0) else eps
ax.set_ylim(min_nonzero * 0.1, valid_data.max() * 10)

ax.set_xlabel("Index")
ax.set_ylabel("Relative Error")
ax.grid(True, axis='both', linestyle='--', linewidth=0.62, alpha=0.55, zorder=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
# ax.spines['left'].set_color('black')
# ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')

# ax.legend(loc="lower right", 
#             frameon=True, 
#             fontsize=12, 
#             ncol=2,
#             bbox_to_anchor=(0.96, 0.14),
#             facecolor='white',
#             edgecolor='black',
#             framealpha=1.0,
#             shadow=True,
#             labelspacing=0.09,
#             columnspacing=1.02
#         )
ax.legend(loc="lower right", 
            frameon=True, 
            fontsize=12, 
            ncol=2,
            bbox_to_anchor=(0.96, 0.14),
            facecolor='#1a1a1a',
            edgecolor='white',
            framealpha=1.0,
            shadow=True,
            labelspacing=0.09,
            columnspacing=1.02
        )

plt.tight_layout()
# plt.savefig("Fig3.pdf", dpi=600, bbox_inches="tight", transparent=True)
plt.savefig("Fig3.png", dpi=600, bbox_inches="tight", transparent=True)
plt.show()