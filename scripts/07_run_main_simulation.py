"""
Step 07 — 본 시뮬레이션.

대표 필드 간 전수 쌍(probe_alphas.jsonl) 을 시뮬한다.
이미 결과가 있는 쌍은 skip_if_exists=true 면 스킵. (재실행/이어달리기 지원)

쓰루풋 추정과 ETA 를 progress.json 에 지속적으로 갱신해서 대시보드가 읽는다.

출력:
  data/pair_results.parquet (또는 .jsonl)  — 쌍별 시뮬 결과
    pair_id, cluster_x, cluster_y, field_x, field_y, expr,
    sharpe, fitness, turnover, returns, drawdown, alpha_id, status, similarity, ts
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
        "alpha_id": result.get("alpha_id", "") or "",
        "status": result.get("status", "UNKNOWN"),
    }


def _load_existing(out_path: Path) -> Dict[str, dict]:
    if not out_path.exists():
        return {}
    done = {}
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                done[obj["pair_id"]] = obj
    except Exception:
        return {}
    return done


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    probe_path = resolve_path(config["paths"]["probe_alphas_file"])
    out_path = resolve_path(config["paths"]["pair_results_file"]).with_suffix(".jsonl")
    ensure_parent(out_path)

    if not probe_path.exists():
        raise RuntimeError("probe_alphas.jsonl 없음. 05 단계를 먼저 실행하세요.")

    # Step 4 결과 확인
    step4_path = resolve_path(config["paths"]["step4_results_file"])
    if step4_path.exists():
        try:
            s4 = json.loads(step4_path.read_text(encoding="utf-8"))
            verdict = s4.get("verdict")
            log_line(STEP, f"step4 verdict={verdict}")
            if verdict not in ("PASS",):
                print(f"\n[경고] step4 verdict={verdict} — {s4.get('reasoning', '')}")
                print("계속 진행하려면 --force 옵션이 필요합니다 (현재 미구현). 중단합니다.")
                update_step(config, STEP, status="failed",
                            message=f"step4 not pass: {verdict}")
                return
        except Exception as e:
            log_line(STEP, f"step4 결과 로드 실패: {e}")
    else:
        log_line(STEP, "step4 결과 없음 — 먼저 step4 를 실행하세요.")
        update_step(config, STEP, status="failed", message="step4 결과 없음")
        return

    # 쌍 로드
    pairs = []
    with open(probe_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    max_pairs = int(config["main_sim"].get("max_pairs", 5000))
    if len(pairs) > max_pairs:
        log_line(STEP, f"쌍 {len(pairs)}개 > max_pairs {max_pairs} → 자름")
        pairs = pairs[:max_pairs]

    # 기존 결과 로드 (이어달리기)
    skip_if_exists = bool(config["main_sim"].get("skip_if_exists", True))
    existing = _load_existing(out_path) if skip_if_exists else {}
    log_line(STEP, f"기존 결과 {len(existing)} 개 (skip_if_exists={skip_if_exists})")

    todo = [p for p in pairs if p["pair_id"] not in existing]
    total = len(pairs)
    skipped = len(pairs) - len(todo)
    log_line(STEP, f"전체 {total} 쌍 중 {len(todo)} 신규, {skipped} 스킵")

    # 시뮬 클라이언트
    client = make_wqb_client(config)
    settings = build_wqb_settings(client, config)

    # 결과는 append 모드로 기록
    f_out = open(out_path, "a", encoding="utf-8")

    # 쓰루풋/ETA 계산용
    started = time.time()
    completed_in_run = 0
    passed = 0
    failed = 0

    for idx, pair in enumerate(todo, 1):
        expr = pair["expr"]
        try:
            result = client.simulate_expression(
                expr, settings,
                max_wait_sec=int(config["wqb"].get("max_wait_sec", 600)),
            )
            m = _extract_metrics(result)
        except Exception as e:
            log_line(STEP, f"err {pair['pair_id']}: {str(e)[:120]}")
            m = {"sharpe": 0.0, "fitness": 0.0, "turnover": 0.0, "returns": 0.0,
                 "drawdown": 0.0, "alpha_id": "", "status": "ERROR", "error": str(e)[:200]}
            failed += 1
        else:
            if m["status"] == "COMPLETE":
                passed += 1
            else:
                failed += 1

        # subtract probe 기준 유사도 변환: |sharpe| 낮을수록 유사
        similarity = max(0.0, 1.0 - min(abs(m["sharpe"]) / 2.0, 1.0))

        record = {
            "pair_id": pair["pair_id"],
            "cluster_x": pair["cluster_x"],
            "cluster_y": pair["cluster_y"],
            "field_x": pair["field_x"],
            "field_y": pair["field_y"],
            "expr": expr,
            **m,
            "similarity": similarity,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_out.flush()

        completed_in_run += 1
        global_completed = len(existing) + completed_in_run
        elapsed = time.time() - started
        tph = (completed_in_run / elapsed) * 3600.0 if elapsed > 0 else 0.0
        remaining = len(todo) - idx
        eta_h = (remaining / (tph / 3600.0)) / 3600.0 if tph > 0 else 0.0

        update_step(
            config, STEP,
            progress_value=global_completed / max(1, total),
            message=f"{global_completed}/{total}  sim={completed_in_run}  eta={eta_h:.1f}h",
            sim_stats={
                "total_planned": total,
                "completed": global_completed,
                "skipped": skipped,
                "passed": passed,
                "failed": failed,
                "current_pair": f"{pair['field_x']} x {pair['field_y']}",
                "throughput_per_hour": round(tph, 1),
                "eta_hours": round(eta_h, 1),
                "phase": "main_sim",
            },
        )

    f_out.close()
    log_line(STEP, f"완료: 신규 {completed_in_run}, 전체 {len(existing) + completed_in_run}/{total}")
    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"{len(existing) + completed_in_run}/{total} 완료",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
