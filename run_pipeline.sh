#!/bin/bash
# Field Graph 파이프라인 전체 실행
#
# 단계:
#   01 corpus → 02 embeddings → 03 cluster → 04 reps → 05 probes
#   → 06 step4 (가정 검증) → 07 main_sim → 08 graph → 09 visualize
#
# 06 에서 verdict != PASS 면 07 는 스킵된다 (스크립트 내부에서 체크).
#
# 환경변수:
#   SKIP_UNTIL=07   → 07 단계부터 실행 (재개)
#   ONLY=06         → 06 만 실행

set -e
cd "$(dirname "$0")"

STEPS=(
  "01_build_corpus.py"
  "02_build_embeddings.py"
  "03_cluster_fields.py"
  "04_pick_representatives.py"
  "05_build_probe_alphas.py"
  "06_step4_validate.py"
  "07_run_main_simulation.py"
  "08_build_graph.py"
  "09_visualize_graph.py"
)

should_run() {
  local step_num="$1"
  if [ -n "$ONLY" ]; then
    [ "$ONLY" = "$step_num" ] && return 0 || return 1
  fi
  if [ -n "$SKIP_UNTIL" ]; then
    [ "$step_num" \< "$SKIP_UNTIL" ] && return 1 || return 0
  fi
  return 0
}

mkdir -p data output logs

LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "=== Field Graph Pipeline ==="
echo "Log: $LOG_FILE"
echo ""

for script in "${STEPS[@]}"; do
  num="${script:0:2}"
  if ! should_run "$num"; then
    echo "[skip] $script"
    continue
  fi
  echo ""
  echo "=== Running: $script ==="
  if ! python3 "scripts/$script" 2>&1 | tee -a "$LOG_FILE"; then
    echo ""
    echo "[error] $script 실패 — 파이프라인 중단"
    echo "로그: $LOG_FILE"
    exit 1
  fi
done

echo ""
echo "=== Pipeline 완료 ==="
echo "결과: output/field_graph.html 를 브라우저에서 여세요"
