"""
Step 09c — Seaborn clustermap PNG (모노크롬).

similarity_combined 매트릭스를 hierarchical clustering 으로 재정렬해서
고해상도 흑백 heatmap 으로 저장.

출력:
  output/site/views/heatmap.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, log_line, update_step,
)

STEP = "09_visualize"


def main():
    config = load_config()
    log_line("09c", "start")

    data_dir = resolve_path(config["paths"]["data_dir"])
    sim_path = data_dir / "similarity_combined.npy"
    out_png = resolve_path("output/site/views/heatmap.png")
    ensure_parent(out_png)

    combined = np.load(sim_path).astype(np.float32)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "axes.facecolor": "#0a0a0a",
        "figure.facecolor": "#0a0a0a",
        "savefig.facecolor": "#0a0a0a",
        "text.color": "#a8a8a8",
        "xtick.color": "#6b6b6b",
        "ytick.color": "#6b6b6b",
        "axes.edgecolor": "#2a2a2a",
        "axes.labelcolor": "#a8a8a8",
    })

    # seaborn clustermap: 거리 = 1 - sim
    distance = 1.0 - combined
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    distance = (distance + distance.T) / 2

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")

    g = sns.clustermap(
        combined,
        row_linkage=Z,
        col_linkage=Z,
        cmap="gray_r",   # 흑백 (1 = 검정, 0 = 흰색 역순: gray_r 이 redundant=밝게)
        vmin=0.0, vmax=1.0,
        figsize=(14, 14),
        xticklabels=False,
        yticklabels=False,
        cbar_pos=(0.02, 0.85, 0.025, 0.10),
        dendrogram_ratio=0.08,
    )
    g.ax_heatmap.set_facecolor("#0a0a0a")
    g.fig.set_facecolor("#0a0a0a")
    # 덴드로그램 색
    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        if ax is None:
            continue
        for line in ax.collections:
            line.set_color("#6b6b6b")
        for line in ax.lines:
            line.set_color("#6b6b6b")

    # colorbar 테두리
    g.cax.set_facecolor("#0a0a0a")
    g.cax.tick_params(colors="#a8a8a8")
    g.cax.yaxis.set_tick_params(color="#2a2a2a")

    g.fig.savefig(str(out_png), dpi=180, bbox_inches="tight",
                  facecolor="#0a0a0a")
    plt.close(g.fig)

    log_line("09c", f"saved → {out_png}")
    update_step(config, STEP, progress_value=0.8,
                message="09c heatmap done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
