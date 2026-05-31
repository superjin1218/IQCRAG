#!/usr/bin/env bash
# IQCRAG v2 pipeline — corr → cluster → umap → site assets
# Assumes 시뮬/output/{pnl_wide,fields_summary}.csv already produced.
#
# Usage (from IQCRAG/ root):
#   bash run_v2_pipeline.sh [SIM_OUTPUT_DIR] [SITE_DIR]
set -euo pipefail
cd "$(dirname "$0")"

SIM_OUT="${1:-/home/jinwoo/Desktop/시뮬/output}"
SITE_DIR="${2:-../IQC-SITE}"
CATALOG="${3:-/home/jinwoo/Desktop/시뮬/all_fields_USA_TOP3000_delay1.csv}"

echo "[v2] SIM_OUT  = $SIM_OUT"
echo "[v2] SITE_DIR = $SITE_DIR"

mkdir -p data_v2

echo "[v2] 11) PnL correlation matrix"
python3 scripts/11_build_pnl_corr_v2.py \
  --pnl "$SIM_OUT/pnl_wide.csv" \
  --summary "$SIM_OUT/fields_summary.csv" \
  --out-dir data_v2

echo "[v2] 12) HDBSCAN clustering"
python3 scripts/12_cluster_v2.py --in-dir data_v2

echo "[v2] 13) UMAP 2D"
python3 scripts/13_umap_v2.py --in-dir data_v2

echo "[v2] 14) Site assets → $SITE_DIR/assets/data/"
python3 scripts/14_build_site_assets_v2.py \
  --in-dir data_v2 \
  --summary "$SIM_OUT/fields_summary.csv" \
  --catalog "$CATALOG" \
  --pnl-wide "$SIM_OUT/pnl_wide.csv" \
  --out-dir "$SITE_DIR/assets/data"

echo "[v2] ✓ done. assets ready in $SITE_DIR/assets/data/"
