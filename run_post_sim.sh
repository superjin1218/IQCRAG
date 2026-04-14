#!/bin/bash
# Step 07b 완료 후 후처리 체이닝 실행
#
# 순서:
#   08a  similarity matrix (cross-sectional demean + combined)
#   08b  behavior recluster (HDBSCAN on combined distance)
#   08   NetworkX graph 빌드
#   09a  PyVis monochrome graph HTML
#   09b  Plotly UMAP map HTML
#   09c  Seaborn heatmap PNG
#   09d  site assembly (index.html, CSS, JS)

set -e
cd "$(dirname "$0")"

LOG="logs/post_sim_$(date +%Y%m%d_%H%M%S).log"
echo "=== Field Graph post-sim chain ===" | tee -a "$LOG"
echo "Log: $LOG"

for script in \
    08a_build_similarity_matrix.py \
    08b_behavior_recluster.py \
    08_build_graph.py \
    09a_build_graph_html.py \
    09b_build_map_html.py \
    09c_build_heatmap_png.py \
    09d_build_site.py
do
    echo "" | tee -a "$LOG"
    echo "=== Running: $script ===" | tee -a "$LOG"
    python3 "scripts/$script" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Done. ===" | tee -a "$LOG"
echo "결과 사이트: output/site/index.html" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "로컬에서 보려면:" | tee -a "$LOG"
echo "  cd output/site && python3 -m http.server 8080" | tee -a "$LOG"
echo "  브라우저에서 http://localhost:8080" | tee -a "$LOG"
