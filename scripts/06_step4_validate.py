"""
Step 06 — 가정 검증 (Step 4, 핵심).

가정: "같은 클러스터 = 유사한 필드" 가 실제 시뮬에서도 성립하는가?

방법:
  1) 같은 클러스터 내부 쌍 50개 샘플 → probe 시뮬 → |sharpe| 분포
  2) 다른 클러스터 쌍 50개 샘플 → probe 시뮬 → |sharpe| 분포
  3) 내부 평균 vs 외부 평균 비교

Probe template = subtract 기준:
  - 유사한 두 필드 (rank 차이 ≈ 0) → 잔여 신호 약함 → |sharpe| 낮음
  - 독립된 두 필드 (rank 차이 큼)   → 잔여 신호 강함 → |sharpe| 높음
  → 유사도 점수 = 1 - normalized(|sharpe|)
    intra 평균 유사도가 높고 inter 평균 유사도가 낮아야 통과.

출력:
  data/step4_results.json
    {
      "template": "subtract",
      "intra": {"samples": [...], "mean_abs_sharpe": 0.12, "mean_similarity": 0.78},
      "inter": {"samples": [...], "mean_abs_sharpe": 0.41, "mean_similarity": 0.22},
      "verdict": "PASS",
      "reasoning": "..."
    }
"""
from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
    make_wqb_client, build_wqb_settings,
)

STEP = "06_step4"


def _sample_intra_pairs(cluster_map: Dict[str, List[str]], k: int, rng: random.Random) -> List[Tuple[str, str, str]]:
    """같은 클러스터 내부 쌍 k 개 샘플. 반환: (cluster_id, field_x, field_y)"""
    eligible = [(cid, fs) for cid, fs in cluster_map.items() if cid != "-1" and len(fs) >= 2]
    if not eligible:
        return []
    out = []
    seen = set()
    attempts = 0
    while len(out) < k and attempts < k * 20:
        attempts += 1
        cid, fs = rng.choice(eligible)
        a, b = rng.sample(fs, 2)
        key = (cid, *sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        out.append((cid, a, b))
    return out


def _sample_inter_pairs(cluster_map: Dict[str, List[str]], k: int, rng: random.Random) -> List[Tuple[str, str, str, str]]:
    """다른 클러스터 간 쌍 k 개. 반환: (cluster_a, cluster_b, field_x, field_y)"""
    cids = [cid for cid in cluster_map.keys() if cid != "-1" and cluster_map[cid]]
    if len(cids) < 2:
        return []
    out = []
    seen = set()
    attempts = 0
    while len(out) < k and attempts < k * 20:
        attempts += 1
        ca, cb = rng.sample(cids, 2)
        fa = rng.choice(cluster_map[ca])
        fb = rng.choice(cluster_map[cb])
        key = tuple(sorted([fa, fb]))
        if key in seen:
            continue
        seen.add(key)
        out.append((ca, cb, fa, fb))
    return out


def _build_expr(config, x: str, y: str) -> str:
    template = config["probe"].get("template", "subtract")
    if template == "subtract":
        return config["probe"]["subtract_expr"].format(x=x, y=y)
    return config["probe"]["add_expr"].format(x=x, y=y)


def _extract_sharpe(result) -> float:
    """WQB 응답에서 sharpe 추출. run_batch_sim.py 와 동일 로직."""
    alpha = result.get("alpha") or {}
    alpha_is = alpha.get("is", {}) if isinstance(alpha, dict) else {}
    if not alpha_is:
        sim = result.get("simulation") or {}
        alpha_is = sim.get("is", {}) if isinstance(sim, dict) else {}
    try:
        return float(alpha_is.get("sharpe") or 0.0)
    except Exception:
        return 0.0


def _run_pair(client, settings, config, x: str, y: str) -> dict:
    """단일 쌍에 대해 probe alpha 시뮬."""
    expr = _build_expr(config, x, y)
    try:
        result = client.simulate_expression(
            expr, settings, max_wait_sec=int(config["wqb"].get("max_wait_sec", 600))
        )
        sharpe = _extract_sharpe(result)
        status = result.get("status", "UNKNOWN")
        alpha_id = result.get("alpha_id", "")
        return {"expr": expr, "sharpe": sharpe, "abs_sharpe": abs(sharpe),
                "status": status, "alpha_id": alpha_id}
    except Exception as e:
        return {"expr": expr, "sharpe": 0.0, "abs_sharpe": 0.0,
                "status": "ERROR", "error": str(e)[:200]}


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="샘플링")
    log_line(STEP, "start")

    clusters_path = resolve_path(config["paths"]["clusters_file"])
    out_path = resolve_path(config["paths"]["step4_results_file"])
    ensure_parent(out_path)

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    cluster_map: Dict[str, List[str]] = clusters["clusters"]

    k_intra = int(config["step4"].get("intra_samples", 50))
    k_inter = int(config["step4"].get("inter_samples", 50))
    seed = int(config["step4"].get("seed", 42))
    rng = random.Random(seed)

    intra_pairs = _sample_intra_pairs(cluster_map, k_intra, rng)
    inter_pairs = _sample_inter_pairs(cluster_map, k_inter, rng)
    log_line(STEP, f"샘플: intra={len(intra_pairs)}, inter={len(inter_pairs)}")

    # 시뮬 클라이언트
    client = make_wqb_client(config)
    settings = build_wqb_settings(client, config)
    log_line(STEP, "WQB 인증 완료")

    total = len(intra_pairs) + len(inter_pairs)
    done = 0
    intra_results = []
    inter_results = []

    update_step(config, STEP, progress_value=0.02,
                message=f"시뮬 시작 (총 {total} 쌍)",
                sim_stats={"total_planned": total, "completed": 0, "phase": "step4"})

    for cid, a, b in intra_pairs:
        r = _run_pair(client, settings, config, a, b)
        r.update({"cluster": cid, "x": a, "y": b, "kind": "intra"})
        intra_results.append(r)
        done += 1
        update_step(
            config, STEP,
            progress_value=0.02 + 0.96 * (done / total),
            message=f"intra {len(intra_results)}/{len(intra_pairs)}",
            sim_stats={"total_planned": total, "completed": done,
                       "current_pair": f"{a} x {b}", "phase": "step4"},
        )

    for ca, cb, a, b in inter_pairs:
        r = _run_pair(client, settings, config, a, b)
        r.update({"cluster_a": ca, "cluster_b": cb, "x": a, "y": b, "kind": "inter"})
        inter_results.append(r)
        done += 1
        update_step(
            config, STEP,
            progress_value=0.02 + 0.96 * (done / total),
            message=f"inter {len(inter_results)}/{len(inter_pairs)}",
            sim_stats={"total_planned": total, "completed": done,
                       "current_pair": f"{a} x {b}", "phase": "step4"},
        )

    # 유사도 환산: subtract probe 기준 → |sharpe| 낮을수록 유사
    def _stats(arr: List[dict]) -> dict:
        if not arr:
            return {"count": 0, "mean_abs_sharpe": 0.0, "mean_similarity": 0.0}
        abs_s = [x["abs_sharpe"] for x in arr]
        mean_as = sum(abs_s) / len(abs_s)
        # 간단 정규화: sharpe 2 를 상한으로 가정 → 유사도 = 1 - min(|s|/2, 1)
        sims = [max(0.0, 1.0 - min(a / 2.0, 1.0)) for a in abs_s]
        mean_sim = sum(sims) / len(sims)
        return {"count": len(arr), "mean_abs_sharpe": mean_as, "mean_similarity": mean_sim}

    intra_stats = _stats(intra_results)
    inter_stats = _stats(inter_results)

    min_intra = float(config["step4"].get("min_intra_similarity", 0.5))
    max_inter = float(config["step4"].get("max_inter_similarity", 0.3))

    intra_ok = intra_stats["mean_similarity"] >= min_intra
    inter_ok = inter_stats["mean_similarity"] <= max_inter
    if intra_ok and inter_ok:
        verdict = "PASS"
        reasoning = "내부 응집력과 외부 분리 모두 기준 충족"
    elif not intra_ok and inter_ok:
        verdict = "INTRA_WEAK"
        reasoning = "내부 응집력 부족. 클러스터 수를 늘려 재클러스터링 권장."
    elif intra_ok and not inter_ok:
        verdict = "INTER_HIGH"
        reasoning = "외부 분리 부족. 임베딩만으론 부족 — dataset/category 반영 필요."
    else:
        verdict = "FAIL"
        reasoning = "임베딩 기반 가정 자체가 깨짐. 전략 재검토 필요."

    payload = {
        "template": config["probe"].get("template", "subtract"),
        "intra_pairs_sampled": len(intra_pairs),
        "inter_pairs_sampled": len(inter_pairs),
        "intra": {**intra_stats, "samples": intra_results},
        "inter": {**inter_stats, "samples": inter_results},
        "thresholds": {"min_intra_similarity": min_intra, "max_inter_similarity": max_inter},
        "verdict": verdict,
        "reasoning": reasoning,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(STEP, f"verdict={verdict}  intra_sim={intra_stats['mean_similarity']:.3f}  inter_sim={inter_stats['mean_similarity']:.3f}")

    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"{verdict} (intra={intra_stats['mean_similarity']:.2f}, inter={inter_stats['mean_similarity']:.2f})",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
