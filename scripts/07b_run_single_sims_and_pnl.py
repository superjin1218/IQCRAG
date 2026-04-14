"""
Step 07b — 단일 필드 알파 시뮬 + daily-pnl 수집 (순차).

각 probe 알파마다:
  1) simulate_expression(rank(ts_backfill(X, 20)), ...)  → alpha_id, sharpe 등
  2) GET /alphas/{alpha_id}/recordsets/daily-pnl         → (T, ) 일별 PnL 벡터
  3) parquet 에 append (field_id, date, pnl 형식)

이어달리기 지원: 이미 완료된 field_id 는 skip.
진행률: progress.json 에 per-field 업데이트 (대시보드 실시간 읽기).

출력:
  data/single_field_meta.jsonl  — 필드별 시뮬 메타 (append)
  data/pnl_records.jsonl        — 필드별 일별 PnL 데이터 (append, long format)
  data/failures.jsonl           — 실패 필드 기록
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
    make_wqb_client, build_wqb_settings,
)

STEP = "07_main_sim"


def _extract_metrics(result) -> dict:
    alpha = result.get("alpha") or {}
    alpha_is = alpha.get("is", {}) if isinstance(alpha, dict) else {}
    if not alpha_is:
        sim = result.get("simulation") or {}
        alpha_is = sim.get("is", {}) if isinstance(sim, dict) else {}

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    return {
        "sharpe": f(alpha_is.get("sharpe")),
        "fitness": f(alpha_is.get("fitness")),
        "turnover": f(alpha_is.get("turnover")),
        "returns": f(alpha_is.get("returns")),
        "drawdown": f(alpha_is.get("drawdown")),
        "longCount": int(alpha_is.get("longCount") or 0),
        "shortCount": int(alpha_is.get("shortCount") or 0),
        "alpha_id": result.get("alpha_id", "") or "",
        "status": result.get("status", "UNKNOWN"),
    }


def _fetch_daily_pnl(client, alpha_id: str,
                     max_retries: int = 6, base_delay: float = 3.0) -> list[tuple[str, float]]:
    """GET /alphas/{id}/recordsets/daily-pnl → [(date_str, pnl_float), ...]

    시뮬 완료 직후엔 recordset 이 아직 생성 안 됐을 수 있어 retry 가 필요하다.
    빈 response 면 backoff 하고 재시도.
    """
    if not alpha_id:
        return []
    url = f"{client.base_url}/alphas/{alpha_id}/recordsets/daily-pnl"
    delay = base_delay
    for attempt in range(max_retries):
        try:
            r = client.session.get(url, timeout=60)
            if r.status_code == 200 and r.content:
                try:
                    j = r.json()
                    recs = j.get("records") or []
                    if recs:
                        return [(str(rec[0]), float(rec[1])) for rec in recs if len(rec) >= 2]
                except Exception:
                    pass
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay = min(delay * 1.5, 20.0)
    return []


def _load_completed(meta_path: Path) -> set[str]:
    if not meta_path.exists():
        return set()
    done = set()
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["field_id"])
                except Exception:
                    continue
    return done


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard index: '0/2' means take even items, '1/2' means odd items")
    args = parser.parse_args()

    shard_idx, shard_total = None, None
    if args.shard:
        shard_idx, shard_total = map(int, args.shard.split("/"))

    config = load_config()
    shard_label = f" [shard {shard_idx}/{shard_total}]" if shard_idx is not None else ""
    update_step(config, STEP, status="running", progress_value=0.0, message=f"시작{shard_label}")
    log_line(STEP, f"start{shard_label}")

    probe_path = resolve_path(config["paths"]["probe_alphas_file"])
    data_dir = resolve_path(config["paths"]["data_dir"])
    # shard별 출력 파일 분리 (충돌 방지)
    suffix = f"_s{shard_idx}" if shard_idx is not None else ""
    meta_path = data_dir / f"single_field_meta{suffix}.jsonl"
    pnl_path = data_dir / f"pnl_records{suffix}.jsonl"
    fail_path = data_dir / f"failures{suffix}.jsonl"
    ensure_parent(meta_path)

    if not probe_path.exists():
        raise RuntimeError("probe_alphas 없음. 05b 먼저 실행.")

    # 이어달리기: 완료된 필드 로드 (원본 + 자기 shard 파일 모두 확인)
    completed = _load_completed(meta_path)
    if shard_idx is not None:
        original_meta = data_dir / "single_field_meta.jsonl"
        if original_meta.exists():
            completed |= _load_completed(original_meta)
        for i in range(shard_total):
            other = data_dir / f"single_field_meta_s{i}.jsonl"
            if other.exists() and other != meta_path:
                completed |= _load_completed(other)
    log_line(STEP, f"이미 완료: {len(completed)}")

    # 전체 probe 로드
    probes = []
    with open(probe_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                probes.append(json.loads(line))
    total = len(probes)
    todo = [p for p in probes if p["field_id"] not in completed]
    if shard_idx is not None:
        todo = [p for i, p in enumerate(todo) if i % shard_total == shard_idx]
    log_line(STEP, f"전체 {total} / 신규 {len(todo)}{shard_label}")

    # 시뮬 클라이언트
    client = make_wqb_client(config)
    settings = build_wqb_settings(client, config)
    log_line(STEP, "WQB 인증 완료")

    # 출력 파일 append 모드
    f_meta = open(meta_path, "a", encoding="utf-8")
    f_pnl = open(pnl_path, "a", encoding="utf-8")
    f_fail = open(fail_path, "a", encoding="utf-8")

    started = time.time()
    run_done = 0
    n_pass = 0
    n_fail = 0
    n_pnl_ok = 0

    for idx, p in enumerate(todo, 1):
        fid = p["field_id"]
        expr = p["expr"]

        # 1. 시뮬
        try:
            result = client.simulate_expression(
                expr, settings,
                max_wait_sec=int(config["wqb"].get("max_wait_sec", 600)),
            )
            m = _extract_metrics(result)
        except Exception as e:
            err = str(e)[:200]
            log_line(STEP, f"sim_err {fid}: {err[:80]}")
            f_fail.write(json.dumps({"field_id": fid, "phase": "sim", "error": err,
                                     "ts": datetime.now().isoformat(timespec="seconds")},
                                    ensure_ascii=False) + "\n")
            f_fail.flush()
            n_fail += 1
            run_done += 1
            continue

        alpha_id = m.get("alpha_id", "")
        status = m.get("status", "")

        # 2. daily-pnl 다운로드 (시뮬 완료된 경우만)
        pnl_records: list[tuple[str, float]] = []
        if alpha_id and status == "COMPLETE":
            pnl_records = _fetch_daily_pnl(client, alpha_id)
            if pnl_records:
                n_pnl_ok += 1

        # 3. 메타 append
        meta_rec = {
            "field_id": fid,
            "expr": expr,
            **m,
            "pnl_days": len(pnl_records),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        f_meta.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")
        f_meta.flush()

        # 4. PnL 레코드 append (long format: one line = one day)
        for dt, v in pnl_records:
            f_pnl.write(json.dumps({"field_id": fid, "date": dt, "pnl": v},
                                   ensure_ascii=False) + "\n")
        f_pnl.flush()

        if status == "COMPLETE":
            n_pass += 1
        else:
            n_fail += 1

        run_done += 1
        global_done = len(completed) + run_done
        elapsed = time.time() - started
        tph = run_done / elapsed * 3600 if elapsed > 0 else 0
        remaining = total - global_done
        eta_h = remaining / tph if tph > 0 else 0

        update_step(
            config, STEP,
            progress_value=global_done / max(1, total),
            message=f"{global_done}/{total}  run={run_done}  pnl_ok={n_pnl_ok}  eta={eta_h:.1f}h",
            sim_stats={
                "total_planned": total,
                "completed": global_done,
                "passed": n_pass,
                "failed": n_fail,
                "pnl_fetched": n_pnl_ok,
                "current_pair": fid,
                "throughput_per_hour": round(tph, 1),
                "eta_hours": round(eta_h, 1),
                "phase": "854_single_sim",
            },
        )

    f_meta.close()
    f_pnl.close()
    f_fail.close()

    log_line(STEP, f"완료: 신규 {run_done}, 전체 {len(completed) + run_done}/{total}, "
                   f"pass={n_pass}, fail={n_fail}, pnl_ok={n_pnl_ok}")
    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"{len(completed) + run_done}/{total} 완료, pnl_ok={n_pnl_ok}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
