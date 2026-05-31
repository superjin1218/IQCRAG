"""
Step 07b — 단일 필드 알파 시뮬 + daily-pnl 수집 (v2: 멀티계정 멀티샤드).

각 probe 알파마다:
  1) simulate_expression(expr, settings) → alpha_id, sharpe 등
  2) GET /alphas/{alpha_id}/recordsets/daily-pnl → 일별 PnL
  3) JSONL append (field_id, date, pnl)

CLI:
  --account <id>     # config.accounts.list 의 id (a1, a2 …). 미지정 시 레거시 단일계정 모드.
  --shard i/N        # 계정 내 샤드 인덱스 (N == max_concurrent).

probe_alphas.jsonl 의 각 행에 미리 account_id + shard_idx 가 박혀 있으므로,
워커는 자기 (account, shard) 셀에 해당하는 필드만 처리한다.

출력 (멀티계정 모드):
  data/single_field_meta_{account}_s{shard}.jsonl
  data/pnl_records_{account}_s{shard}.jsonl
  data/failures_{account}_s{shard}.jsonl
  data/settings_used.json (1회만, race-safe)

이어달리기: data/single_field_meta_*.jsonl glob 전체 합집합으로 완료 필드 판정.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    빈 response 면 backoff 재시도.
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


def _load_completed_from(meta_path: Path) -> set[str]:
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


def _load_all_completed(data_dir: Path) -> set[str]:
    """data/single_field_meta*.jsonl 전체 glob 합집합."""
    completed: set[str] = set()
    for p in data_dir.glob("single_field_meta*.jsonl"):
        completed |= _load_completed_from(p)
    return completed


def _ensure_settings_snapshot(config: dict, settings: dict, data_dir: Path) -> str:
    """data/settings_used.json 1회 생성 (race-safe). 파일명 반환."""
    snap_name = "settings_used.json"
    snap_path = data_dir / snap_name
    if snap_path.exists():
        return snap_name
    snap = {
        "settings": settings,
        "backfill_window": int(config["probe"]["backfill_window"]),
        "wqb": {k: config["wqb"].get(k) for k in
                ("region", "universe", "delay", "decay", "neutralization", "truncation", "nan_handling")},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(snap_path, "x", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)
    except FileExistsError:
        # 다른 워커가 먼저 생성. OK.
        pass
    return snap_name


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=str, default=None,
                        help="config.accounts.list 의 id (예: a1). 미지정 시 레거시 단일계정.")
    parser.add_argument("--shard", type=str, default=None,
                        help="샤드 'i/N' (계정 내 인덱스. N == max_concurrent)")
    args = parser.parse_args()

    shard_idx, shard_total = None, None
    if args.shard:
        shard_idx, shard_total = map(int, args.shard.split("/"))

    config = load_config()
    accounts_cfg_present = bool(config.get("accounts", {}).get("list"))
    if accounts_cfg_present and not args.account:
        raise SystemExit(
            "config.accounts.list 가 있는데 --account 가 지정되지 않았습니다.\n"
            "예: python 07b_run_single_sims_and_pnl.py --account a1 --shard 0/3"
        )

    label_parts = []
    if args.account:
        label_parts.append(args.account)
    if shard_idx is not None:
        label_parts.append(f"s{shard_idx}/{shard_total}")
    label = " [" + " ".join(label_parts) + "]" if label_parts else ""

    update_step(config, STEP, status="running", progress_value=0.0, message=f"시작{label}")
    log_line(STEP, f"start{label}")

    probe_path = resolve_path(config["paths"]["probe_alphas_file"])
    data_dir = resolve_path(config["paths"]["data_dir"])
    ensure_parent(data_dir / ".keep")

    # 출력 파일명: 멀티계정이면 _{account}_s{shard}, 레거시면 _s{shard}
    if args.account:
        suffix = f"_{args.account}_s{shard_idx}" if shard_idx is not None else f"_{args.account}"
    else:
        suffix = f"_s{shard_idx}" if shard_idx is not None else ""

    meta_path = data_dir / f"single_field_meta{suffix}.jsonl"
    pnl_path = data_dir / f"pnl_records{suffix}.jsonl"
    fail_path = data_dir / f"failures{suffix}.jsonl"

    if not probe_path.exists():
        raise RuntimeError("probe_alphas 없음. 05b 먼저 실행.")

    # 이어달리기: 전체 meta 파일 합집합
    completed = _load_all_completed(data_dir)
    log_line(STEP, f"이미 완료: {len(completed)}")

    # 전체 probe 로드
    probes: list[dict] = []
    with open(probe_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                probes.append(json.loads(line))
    total = len(probes)

    # (account, shard) 필터링
    if args.account:
        if shard_idx is None:
            raise SystemExit("--account 지정 시 --shard i/N 도 필요합니다.")
        todo = [p for p in probes
                if p.get("account_id") == args.account
                and p.get("shard_idx") == shard_idx
                and p["field_id"] not in completed]
    else:
        # 레거시: account 필드 무시, --shard 가 있으면 modulo 분할
        todo = [p for p in probes if p["field_id"] not in completed]
        if shard_idx is not None:
            todo = [p for i, p in enumerate(todo) if i % shard_total == shard_idx]

    my_total = sum(1 for p in probes
                   if (not args.account or
                       (p.get("account_id") == args.account and p.get("shard_idx") == shard_idx)))
    log_line(STEP, f"전체 probe {total} / 내 슬롯 {my_total} / 신규 {len(todo)}{label}")

    # 시뮬 클라이언트 (계정별)
    client = make_wqb_client(config, account_id=args.account)
    settings = build_wqb_settings(client, config)
    log_line(STEP, f"WQB 인증 완료 ({args.account or 'legacy'})")

    # settings 스냅샷
    snap_name = _ensure_settings_snapshot(config, settings, data_dir)

    # 출력 파일 append 모드
    f_meta = open(meta_path, "a", encoding="utf-8")
    f_pnl = open(pnl_path, "a", encoding="utf-8")
    f_fail = open(fail_path, "a", encoding="utf-8")

    # 진행률 step 이름: 멀티계정이면 lane 별로 분리
    if args.account:
        step_key = f"{STEP}_{args.account}_s{shard_idx}"
    else:
        step_key = STEP if shard_idx is None else f"{STEP}_s{shard_idx}"

    started = time.time()
    run_done = 0
    n_pass = 0
    n_fail = 0
    n_pnl_ok = 0

    for idx, p in enumerate(todo, 1):
        fid = p["field_id"]
        expr = p["expr"]
        ftype = p.get("type", "")
        reducer = p.get("reducer", "")

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
            f_fail.write(json.dumps({
                "field_id": fid, "phase": "sim", "error": err,
                "account_id": args.account, "shard_idx": shard_idx,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            f_fail.flush()
            n_fail += 1
            run_done += 1
            continue

        alpha_id = m.get("alpha_id", "")
        status = m.get("status", "")

        # 2. daily-pnl
        pnl_records: list[tuple[str, float]] = []
        if alpha_id and status == "COMPLETE":
            pnl_records = _fetch_daily_pnl(client, alpha_id)
            if pnl_records:
                n_pnl_ok += 1

        # 3. 메타
        meta_rec = {
            "field_id": fid,
            "expr": expr,
            "type": ftype,
            "reducer": reducer,
            "account_id": args.account,
            "shard_idx": shard_idx,
            "settings_snapshot": snap_name,
            **m,
            "pnl_days": len(pnl_records),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        f_meta.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")
        f_meta.flush()

        # 4. PnL long format
        for dt, v in pnl_records:
            f_pnl.write(json.dumps({"field_id": fid, "date": dt, "pnl": v},
                                   ensure_ascii=False) + "\n")
        f_pnl.flush()

        if status == "COMPLETE":
            n_pass += 1
        else:
            n_fail += 1

        run_done += 1
        elapsed = time.time() - started
        tph = run_done / elapsed * 3600 if elapsed > 0 else 0
        remaining = max(0, my_total - (run_done + (my_total - len(todo))))
        eta_h = remaining / tph if tph > 0 else 0

        update_step(
            config, step_key,
            progress_value=(run_done + (my_total - len(todo))) / max(1, my_total),
            message=f"{run_done}/{len(todo)} (slot {my_total}) pnl_ok={n_pnl_ok} eta={eta_h:.1f}h{label}",
            sim_stats={
                "total_planned": total,
                "my_slot_total": my_total,
                "completed_in_slot": run_done + (my_total - len(todo)),
                "passed": n_pass,
                "failed": n_fail,
                "pnl_fetched": n_pnl_ok,
                "current_field": fid,
                "throughput_per_hour": round(tph, 1),
                "eta_hours": round(eta_h, 1),
                "phase": "v2_single_sim",
                "account_id": args.account,
                "shard_idx": shard_idx,
            },
        )

    f_meta.close()
    f_pnl.close()
    f_fail.close()

    log_line(STEP, f"완료{label}: 신규 {run_done}, pass={n_pass}, fail={n_fail}, pnl_ok={n_pnl_ok}")
    update_step(
        config, step_key, status="done", progress_value=1.0,
        message=f"slot 완료 {run_done} (pass={n_pass} fail={n_fail} pnl_ok={n_pnl_ok}){label}",
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
