#!/usr/bin/env bash
# 멀티계정 × 멀티샤드 시뮬 launcher
# config.yaml 의 accounts.list 를 읽어 각 계정마다 max_concurrent 프로세스 fan-out
# 사용: bash run_multi_account.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PIDS=()
cleanup() {
  echo
  echo "[launcher] signal received — killing children: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait
  exit 130
}
trap cleanup INT TERM

# config 에서 (account_id, shard_idx, total_shards) 매핑 추출
mapping=$(python3 - <<'PY'
import yaml, sys
try:
    cfg = yaml.safe_load(open("config.yaml"))
except Exception as e:
    print(f"[launcher-py] config.yaml 로드 실패: {e}", file=sys.stderr)
    sys.exit(1)
accounts = (cfg.get("accounts") or {}).get("list") or []
if not accounts:
    print("[launcher-py] config.accounts.list 가 비어있음", file=sys.stderr)
    sys.exit(1)
default_n = (cfg.get("accounts") or {}).get("default_max_concurrent", 3)
for a in accounts:
    n = int(a.get("max_concurrent", default_n))
    for s in range(n):
        print(f"{a['id']} {s} {n}")
PY
)
if [ -z "$mapping" ]; then
  echo "[launcher] 매핑 생성 실패. config.yaml 확인 필요." >&2
  exit 1
fi

ts=$(date +%Y%m%d_%H%M%S)
echo "[launcher] start ts=${ts}"

while read -r acct shard total; do
  [ -z "$acct" ] && continue
  log="logs/sim_${acct}_s${shard}_${ts}.log"
  echo "[launcher] spawn ${acct} shard ${shard}/${total}  →  ${log}"
  python3 scripts/07b_run_single_sims_and_pnl.py \
      --account "$acct" --shard "${shard}/${total}" \
      >>"$log" 2>&1 &
  PIDS+=($!)
done <<<"$mapping"

echo "[launcher] launched ${#PIDS[@]} workers: pids=${PIDS[*]}"
echo "[launcher] tail -f logs/sim_*_${ts}.log  로 모니터링"
wait
echo "[launcher] all workers finished"
