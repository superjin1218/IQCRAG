"""
Step 14 (v2) — Build IQC-SITE assets from corr/cluster/umap + sim summary.

Input:
  data_v2/pnl_corr.npy
  data_v2/field_ids.json
  data_v2/clusters.json
  data_v2/umap.json
  --summary  : fields_summary.csv (from 시뮬/output/)
  --pnl-wide : pnl_wide.csv (for per-field lazy-loaded PnL series)
  --out-dir  : IQC-SITE/assets/data/

Output (all JSON, compact):
  nodes.json         — [{id, type, reducer, expr, alpha_id, dataset, category,
                          subcategory, desc, sharpe, fitness, turnover, returns,
                          drawdown, longCount, shortCount, x, y}]
  edges.json         — [[i, j, r], ...]   top-K + |r| > 0.25
  neighbors.json     — {id: [{id, r}, ...]} top-10 per node
  umap.json          — [{id, x, y}]
  groups.json        — { by_threshold: {"0.10": [...], ..., "0.50": [...]},
                         cluster_heatmap: { labels, matrix } }
  datasets.json      — [{id, name, count, color}]
  pnl/{field_id}.json — [{d, v}, ...]  (1,236 dates per file)
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd


# A restrained 17-color palette — cartographic, no neon
DATASET_COLORS = {
    "analyst4":      "#c89455",  # copper
    "fundamental2":  "#7d9b6b",  # sage
    "fundamental6":  "#b8553a",  # rust
    "model16":       "#d4a04a",  # amber
    "model51":       "#6b8a90",  # slate
    "model53":       "#8a6b9b",  # mauve
    "model77":       "#a8895c",  # sand
    "news12":        "#5e8a78",  # teal
    "news18":        "#9b6b6b",  # rose
    "option8":       "#7c8aa8",  # steel
    "option9":       "#a08a4a",  # ochre
    "pv1":           "#cc7a52",  # terracotta
    "pv13":          "#6b8a5a",  # moss
    "sentiment1":    "#aa6b7e",  # plum
    "socialmedia12": "#88865c",  # olive
    "socialmedia8":  "#5a7e8a",  # marine
    "univ1":         "#7a7563",  # ink-dim
}
DEFAULT_COLOR = "#9b9080"


def dataset_color(ds: str) -> str:
    if not ds:
        return DEFAULT_COLOR
    return DATASET_COLORS.get(ds, DEFAULT_COLOR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data_v2")
    p.add_argument("--summary", required=True, help="fields_summary.csv (from aggregate)")
    p.add_argument("--catalog", required=True, help="all_fields_*.csv with id/description/dataset_id columns")
    p.add_argument("--pnl-wide", required=True, help="pnl_wide.csv")
    p.add_argument("--out-dir", required=True, help="IQC-SITE/assets/data/")
    p.add_argument("--top-k-neighbors", type=int, default=10)
    p.add_argument("--edge-threshold", type=float, default=0.25)
    p.add_argument("--edges-per-node", type=int, default=12,
                   help="cap edges per node (densest by |r|)")
    p.add_argument("--pnl-stride", type=int, default=1,
                   help=">1 to downsample PnL series for speed")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pnl").mkdir(parents=True, exist_ok=True)

    # ── load core ─────────────────────────────────────────────────────
    corr = np.load(in_dir / "pnl_corr.npy")
    ids = json.loads((in_dir / "field_ids.json").read_text(encoding="utf-8"))
    id2i = {fid: i for i, fid in enumerate(ids)}
    clusters = json.loads((in_dir / "clusters.json").read_text(encoding="utf-8"))
    umap_list = json.loads((in_dir / "umap.json").read_text(encoding="utf-8"))
    umap_by_id = {u["id"]: (u["x"], u["y"]) for u in umap_list}

    # ── load summary + catalog ────────────────────────────────────────
    summary = pd.read_csv(args.summary)
    summary_by_id = {}
    for _, r in summary.iterrows():
        fid = r.get("field_id")
        if not isinstance(fid, str):
            continue
        summary_by_id[fid] = r

    catalog = pd.read_csv(args.catalog)
    # catalog uses 'id' as field id column
    id_col = "id" if "id" in catalog.columns else "field_id"
    catalog_by_id = {}
    for _, r in catalog.iterrows():
        fid = r.get(id_col)
        if not isinstance(fid, str):
            continue
        catalog_by_id[fid] = r

    # ── nodes.json ────────────────────────────────────────────────────
    nodes = []
    dataset_counts = Counter()

    for fid in ids:
        r = summary_by_id.get(fid)
        c = catalog_by_id.get(fid)
        if r is None and c is None:
            continue

        # dataset + description from catalog
        ds = str((c.get("dataset_id") if c is not None else "") or "")
        desc = str((c.get("description") if c is not None else "") or "")
        category = str((c.get("category") if c is not None else "") or "")
        subcategory = str((c.get("subcategory") if c is not None else "") or "")

        dataset_counts[ds] += 1
        x, y = umap_by_id.get(fid, (0.0, 0.0))

        def f(k, default=None):
            if r is None: return default
            v = r.get(k, default)
            try: return float(v)
            except: return default

        def s(k, default=""):
            if r is None: return default
            v = r.get(k, default)
            return str(v) if v is not None and pd.notna(v) else default

        nodes.append({
            "id": fid,
            "type": s("type"),
            "reducer": s("reducer"),
            "expr": s("expr"),
            "alpha_id": s("alpha_id"),
            "dataset": ds,
            "category": category,
            "subcategory": subcategory,
            "desc": desc,
            "sharpe":     f("sharpe"),
            "fitness":    f("fitness"),
            "turnover":   f("turnover"),
            "returns":    f("returns"),
            "drawdown":   f("drawdown"),
            "longCount":  int(r.get("longCount") or 0) if r is not None and pd.notna(r.get("longCount", None)) else None,
            "shortCount": int(r.get("shortCount") or 0) if r is not None and pd.notna(r.get("shortCount", None)) else None,
            "cluster":    clusters.get(fid, -1),
            "x": float(x),
            "y": float(y),
        })

    # write nodes.json
    (out / "nodes.json").write_text(
        json.dumps(nodes, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print(f"[14] nodes.json: {len(nodes):,}")

    # ── edges.json + neighbors.json ───────────────────────────────────
    # For each node, find top edges_per_node neighbors by |r|, then dedup pairs
    # Also build top-K neighbors per node for the detail panel.
    print(f"[14] edges/neighbors threshold |r|>={args.edge_threshold} ...")
    N = corr.shape[0]
    edge_set = set()
    edges = []
    neighbors = {}
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)

    for i, fid_i in enumerate(ids):
        order = np.argsort(-abs_corr[i])
        # neighbors: top-K with |r| >= 0.10 (lower bar than edges)
        nbs = []
        for j in order[: args.top_k_neighbors * 2]:
            if i == j: continue
            r = float(corr[i, j])
            if abs(r) < 0.10: break
            nbs.append({"id": ids[j], "r": round(r, 4)})
            if len(nbs) >= args.top_k_neighbors: break
        neighbors[fid_i] = nbs

        # edges: stricter threshold
        added = 0
        for j in order[: args.edges_per_node * 3]:
            if added >= args.edges_per_node: break
            if j <= i: continue
            r = float(corr[i, j])
            if abs(r) < args.edge_threshold: continue
            pair = (i, int(j))
            if pair in edge_set: continue
            edge_set.add(pair)
            edges.append([fid_i, ids[j], round(r, 4)])
            added += 1

    (out / "edges.json").write_text(
        json.dumps(edges, separators=(",", ":")),
        encoding="utf-8")
    (out / "neighbors.json").write_text(
        json.dumps(neighbors, separators=(",", ":")),
        encoding="utf-8")
    print(f"[14] edges.json: {len(edges):,}  neighbors.json: {len(neighbors):,}")

    # ── umap.json ─────────────────────────────────────────────────────
    (out / "umap.json").write_text(
        json.dumps(umap_list, separators=(",", ":")),
        encoding="utf-8")

    # ── groups.json ───────────────────────────────────────────────────
    # 4-threshold cluster cards + cluster heatmap
    print("[14] building groups ...")
    cluster_members = defaultdict(list)
    for fid, cid in clusters.items():
        cluster_members[int(cid)].append(fid)
    cluster_ids = sorted([c for c in cluster_members if c != -1])

    # cluster heatmap (cluster × cluster mean |r|)
    means = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=np.float32)
    cluster_idx = {c: i for i, c in enumerate(cluster_ids)}
    for ci, ca in enumerate(cluster_ids):
        ia = [id2i[f] for f in cluster_members[ca] if f in id2i]
        for cj, cb in enumerate(cluster_ids):
            ib = [id2i[f] for f in cluster_members[cb] if f in id2i]
            if not ia or not ib: continue
            sub = abs_corr[np.ix_(ia, ib)]
            means[ci, cj] = float(sub.mean())
    cluster_labels = [f"C{c}" for c in cluster_ids]

    # per-threshold cards (largest clusters first)
    def cards_for_threshold(thr: float):
        out_cards = []
        for cid in sorted(cluster_ids, key=lambda c: -len(cluster_members[c])):
            members = cluster_members[cid]
            if len(members) < 3:
                continue
            # mean abs corr within cluster (use as "density" — proxy for threshold)
            idx = [id2i[f] for f in members if f in id2i]
            sub = abs_corr[np.ix_(idx, idx)]
            mean_r = float(sub.mean())
            if mean_r < thr:
                continue
            # mean sharpe + dominant dataset
            shar = np.mean([
                next((n["sharpe"] for n in nodes if n["id"] == f), 0) or 0
                for f in members[:20]
            ])
            ds_count = Counter()
            for f in members:
                node = next((n for n in nodes if n["id"] == f), None)
                if node and node["dataset"]:
                    ds_count[node["dataset"]] += 1
            top_ds = ds_count.most_common(1)[0][0] if ds_count else ""
            # sample fields (highest sharpe first)
            member_sharpe = [
                (f, next((n["sharpe"] for n in nodes if n["id"] == f), 0) or 0)
                for f in members
            ]
            member_sharpe.sort(key=lambda x: -abs(x[1] or 0))
            out_cards.append({
                "id": f"C{cid}",
                "n": len(members),
                "mean_sharpe": float(shar),
                "top_dataset": top_ds,
                "sample_fields": [f for f, _ in member_sharpe[:6]],
                "density": mean_r,
            })
        return out_cards

    groups = {
        "by_threshold": {
            "0.10": cards_for_threshold(0.10),
            "0.25": cards_for_threshold(0.25),
            "0.35": cards_for_threshold(0.35),
            "0.50": cards_for_threshold(0.50),
        },
        "cluster_heatmap": {
            "labels": cluster_labels,
            "matrix": means.tolist(),
        },
    }
    (out / "groups.json").write_text(
        json.dumps(groups, separators=(",", ":")),
        encoding="utf-8")
    print(f"[14] groups.json · clusters={len(cluster_ids)}")

    # ── datasets.json ─────────────────────────────────────────────────
    ds_names = {
        "analyst4":      "Analyst Estimate Data",
        "fundamental2":  "Report Footnotes",
        "fundamental6":  "Company Fundamental Data",
        "model16":       "Fundamental Scores",
        "model51":       "Systematic Risk Metrics",
        "model53":       "Creditworthiness Risk",
        "model77":       "Analysts' Factor Model",
        "news12":        "US News Data",
        "news18":        "Ravenpack News",
        "option8":       "Volatility Data",
        "option9":       "Options Analytics",
        "pv1":           "Price Volume",
        "pv13":          "Relationship Data",
        "sentiment1":    "Research Sentiment",
        "socialmedia12": "Sentiment (Equity)",
        "socialmedia8":  "Social Media (Equity)",
        "univ1":         "Universe",
    }
    datasets = []
    for ds, count in dataset_counts.most_common():
        if not ds:
            continue
        datasets.append({
            "id": ds,
            "name": ds_names.get(ds, ds),
            "count": int(count),
            "color": dataset_color(ds),
        })
    (out / "datasets.json").write_text(
        json.dumps(datasets, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print(f"[14] datasets.json: {len(datasets)}")

    # ── per-field PnL series (lazy-load by detail panel) ──────────────
    print(f"[14] writing per-field PnL series → {out / 'pnl'} ...")
    pnl_df = pd.read_csv(args.pnl_wide, index_col="date")
    dates = list(pnl_df.index)
    if args.pnl_stride > 1:
        dates_keep = dates[::args.pnl_stride]
    else:
        dates_keep = dates

    written = 0
    for fid in ids:
        if fid not in pnl_df.columns:
            continue
        col = pnl_df[fid]
        if args.pnl_stride > 1:
            col = col.iloc[::args.pnl_stride]
        series = [
            {"d": str(d), "v": float(v)}
            for d, v in zip(col.index, col.values)
            if pd.notna(v)
        ]
        if not series:
            continue
        (out / "pnl" / f"{fid}.json").write_text(
            json.dumps(series, separators=(",", ":")),
            encoding="utf-8")
        written += 1
        if written % 500 == 0:
            print(f"     {written}/{len(ids)}")
    print(f"[14] per-field PnL written: {written:,}")

    print("[14] ✓ all assets ready in", out)


if __name__ == "__main__":
    main()
