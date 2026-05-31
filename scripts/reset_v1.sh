#!/usr/bin/env bash
# v1 (backfill=20, prefix-매칭) PnL/유사도 데이터 일괄 삭제.
# 백업은 GitHub 에 있음. 사용: bash scripts/reset_v1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[reset] deleting v1 PnL/similarity data ..."
for f in \
  data/pnl_records.jsonl data/pnl_records_s*.jsonl data/pnl_records_a*_s*.jsonl \
  data/single_field_meta.jsonl data/single_field_meta_s*.jsonl data/single_field_meta_a*_s*.jsonl \
  data/failures.jsonl data/failures_s*.jsonl data/failures_a*_s*.jsonl \
  data/similarity_pnl.npy data/similarity_text.npy data/similarity_combined.npy \
  data/similarity_field_ids.json data/similarity_meta.json \
  data/probe_alphas.jsonl data/probe_alphas_skipped.jsonl \
  data/settings_used.json ; do
  if compgen -G "$f" > /dev/null; then
    rm -v $f
  fi
done
echo "[reset] done"
echo "[reset] keep: field_corpus.jsonl, embeddings.npy, field_ids.json, clusters*.json, representatives.json, umap_coords.json"
