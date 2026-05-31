"""
Step 13 (v2) — UMAP 2D from PnL correlation distance.

Output:
  data_v2/umap.json — [{id, x, y}, ...]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="data_v2")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    corr = np.load(in_dir / "pnl_corr.npy")
    ids = json.loads((in_dir / "field_ids.json").read_text(encoding="utf-8"))

    dist = (1.0 - np.abs(corr)).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0.0, 1.0)

    try:
        import umap
    except ImportError:
        raise SystemExit("umap-learn not installed. pip install umap-learn")

    print(f"[13] UMAP n_neighbors={args.n_neighbors} min_dist={args.min_dist}")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="precomputed",
        random_state=args.seed,
        n_components=2,
    )
    coords = reducer.fit_transform(dist.astype(np.float64))

    out = [{"id": fid, "x": float(coords[i, 0]), "y": float(coords[i, 1])}
           for i, fid in enumerate(ids)]
    (in_dir / "umap.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"[13] {len(out)} points → {in_dir / 'umap.json'}")


if __name__ == "__main__":
    main()
