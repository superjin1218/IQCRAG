"""
Step 12 (v2) — HDBSCAN clustering on PnL correlation distance.

Input:
  data_v2/pnl_corr.npy
  data_v2/field_ids.json

Output:
  data_v2/clusters.json     — {field_id: cluster_id (int, -1 = noise)}
  data_v2/cluster_meta.json — {n_clusters, sizes, params, ...}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="data_v2")
    parser.add_argument("--min-cluster-size", type=int, default=18)
    parser.add_argument("--min-samples", type=int, default=4)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    corr = np.load(in_dir / "pnl_corr.npy")
    ids = json.loads((in_dir / "field_ids.json").read_text(encoding="utf-8"))

    # distance = 1 - |corr|
    dist = (1.0 - np.abs(corr)).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    # symmetrize / clip
    dist = np.clip((dist + dist.T) / 2, 0.0, 1.0)

    try:
        import hdbscan
    except ImportError:
        raise SystemExit("hdbscan not installed. pip install hdbscan")

    print(f"[12] HDBSCAN min_cluster_size={args.min_cluster_size} min_samples={args.min_samples}")
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(dist)

    cluster_of = {fid: int(lab) for fid, lab in zip(ids, labels)}
    (in_dir / "clusters.json").write_text(json.dumps(cluster_of), encoding="utf-8")

    sizes = {}
    for lab in labels:
        sizes[int(lab)] = sizes.get(int(lab), 0) + 1
    n_clusters = sum(1 for k in sizes if k != -1)
    (in_dir / "cluster_meta.json").write_text(json.dumps({
        "n_clusters": n_clusters,
        "n_noise": sizes.get(-1, 0),
        "sizes": sizes,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }, indent=2), encoding="utf-8")

    print(f"[12] {n_clusters} clusters · noise {sizes.get(-1, 0)} / {len(ids)}")


if __name__ == "__main__":
    main()
