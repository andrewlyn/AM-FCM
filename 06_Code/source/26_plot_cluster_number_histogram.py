"""Plot cluster_num histograms at -10 dB (left) and -8 dB (right)."""
from pathlib import Path

from pathlib import Path

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from pathlib import Path

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

data_m10 = {2: 83, 3: 537, 4: 215, 5: 51, 6: 29, 7: 9, 8: 9, 9: 10, 10: 8}

data_m8 = {2: 308, 3: 568, 4: 80, 5: 11, 6: 1, 7: 1, 9: 1, 10: 1}

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["font.size"] = 7
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.top"] = True
rcParams["ytick.right"] = True

bins = list(range(2, 11))  # 2..10

sub_w_in = 86.0 / 25.4
sub_h_in = sub_w_in * 0.75
fig, axes = plt.subplots(1, 2, figsize=(2 * sub_w_in, sub_h_in))

def draw(ax, data, tag, color):
    counts = [data.get(b, 0) for b in bins]
    bars = ax.bar(bins, counts, width=0.75, color=color,
                  edgecolor="black", linewidth=0.8, zorder=3)

    ymax = max(counts)
    for b, c in zip(bins, counts):
        if c > 0:
            ax.text(b, c + ymax * 0.015, str(c),
                    ha="center", va="bottom", fontsize=7, color="black")
    ax.set_xticks(bins)
    ax.set_xlabel("Cluster number", fontsize=7)

    ax.text(0.03, 0.97, tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=7)
    ax.set_ylim(0, ymax * 1.15)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

draw(axes[0], data_m10, "(a)", "#4C72B0")
axes[0].set_ylabel("Count", fontsize=7)
draw(axes[1], data_m8,  "(b)", "#DD8452")

fig.tight_layout()
out_tif = str(Path(__file__).resolve().parent / 'cluster_num_histogram.tif')
out_pdf = out_tif.replace(".tif", ".pdf")
fig.savefig(out_tif, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(out_pdf, bbox_inches="tight")
print("saved:", out_tif)
print("saved:", out_pdf)

print("total -10 dB:", sum(data_m10.values()))
print("total -8 dB :", sum(data_m8.values()))
