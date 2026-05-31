"""
Step 11 (v2) — Build PnL correlation matrix from sim output.

Input:
  --pnl     : path to pnl_wide.csv (date × field, from 시뮬/output/)
  --summary : path to fields_summary.csv
  --out-dir : where to write npy + json (default IQCRAG/data_v2)

Output:
  data_v2/pnl_corr.npy        — (N, N) float32, |Pearson corr| with sign
  data_v2/field_ids.json      — ordered field id list (length N)
  data_v2/corr_meta.json      — { dates, n_dates, n_fields, demeaned, generated_at }

Algorithm:
  1. Load pnl_wide.csv (date index, fields columns)
  2. Drop fields with too few valid days (default min_days=252)
  3. Cross-sectional demean (per row: subtract daily mean across fields)
  4. np.corrcoef → (N, N) Pearson
  5. Save matrix + ordered ids
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnl", required=True, help="path to pnl_wide.csv")
    parser.add_argument("--summary", required=True, help="path to fields_summary.csv")
    parser.add_argument("--out-dir", default="data_v2", help="output directory")
    parser.add_argument("--min-days", type=int, default=252)
    parser.add_argument("--demean", action="store_true", default=True,
                        help="cross-sectional demean per date (default ON)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[11] load {args.pnl}")
    df = pd.read_csv(args.pnl, index_col="date")
    print(f"     shape: {df.shape}  ({df.shape[0]:,} dates × {df.shape[1]:,} fields)")

    # field with at least min_days non-null
    valid_mask = df.notna().sum(axis=0) >= args.min_days
    df = df.loc[:, valid_mask]
    print(f"     after min_days={args.min_days}: {df.shape[1]:,} fields kept")

    # cross-sectional demean
    if args.demean:
        df = df.sub(df.mean(axis=1), axis=0)
        print("     cross-sectional demean: ON")

    # correlation matrix (handles NaN via pairwise mask)
    print("[11] computing pearson corr...")
    mat = df.to_numpy(dtype=np.float32)
    # fillna with 0 after demean (safe because mean removed)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    # column-wise z-score
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    z = (mat - mu) / sd
    corr = (z.T @ z) / mat.shape[0]
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)

    field_ids = list(df.columns)
    np.save(out / "pnl_corr.npy", corr)
    (out / "field_ids.json").write_text(json.dumps(field_ids), encoding="utf-8")
    (out / "corr_meta.json").write_text(json.dumps({
        "n_dates": int(df.shape[0]),
        "n_fields": int(df.shape[1]),
        "dates_first": str(df.index[0]),
        "dates_last": str(df.index[-1]),
        "demeaned": bool(args.demean),
        "min_days": args.min_days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    print(f"[11] saved corr ({corr.shape[0]} × {corr.shape[1]}) → {out}")


if __name__ == "__main__":
    main()
