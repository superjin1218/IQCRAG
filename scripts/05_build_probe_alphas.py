"""
Step 05 — Probe Alpha 수식 빌더.

대표 필드 간 쌍을 만들고, 각 쌍에 대해 probe alpha 수식을 생성한다.
실제 시뮬은 06(step4) / 07(main) 이 한다. 여기서는 수식만 파일로 떨어뜨린다.

출력:
  data/probe_alphas.jsonl
    {"pair_id": "0_1", "field_x": "...", "field_y": "...",
     "cluster_x": "0", "cluster_y": "1", "expr": "subtract(rank(...), rank(...))",
     "same_cluster": false}
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
)

STEP = "05_probe_alphas"


def build_expr(template: str, x: str, y: str, config: dict) -> str:
    if template == "subtract":
        tmpl = config["probe"]["subtract_expr"]
    elif template == "add":
        tmpl = config["probe"]["add_expr"]
    else:
        raise ValueError(f"알 수 없는 probe template: {template}")
    return tmpl.format(x=x, y=y)


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    reps_path = resolve_path(config["paths"]["representatives_file"])
    clusters_path = resolve_path(config["paths"]["clusters_file"])
    out_path = resolve_path(config["paths"]["probe_alphas_file"])
    ensure_parent(out_path)

    reps_data = json.loads(reps_path.read_text(encoding="utf-8"))
    reps: dict[str, dict] = reps_data["representatives"]

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    assignments: dict[str, int] = clusters["assignments"]

    template = config["probe"].get("template", "subtract")

    # 쌍 생성: 대표 필드끼리 C(n, 2)
    rep_items = sorted(reps.items(), key=lambda kv: int(kv[0]))
    rep_list = [(cid, info["field_id"]) for cid, info in rep_items]

    pairs_out = []
    for (cid_a, fa), (cid_b, fb) in combinations(rep_list, 2):
        pair = {
            "pair_id": f"{cid_a}_{cid_b}",
            "cluster_x": cid_a,
            "cluster_y": cid_b,
            "field_x": fa,
            "field_y": fb,
            "expr": build_expr(template, fa, fb, config),
            "same_cluster": False,
        }
        pairs_out.append(pair)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log_line(STEP, f"대표 간 쌍 {len(pairs_out)} 개 생성 → {out_path}")
    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"{len(pairs_out)} 쌍 생성 (template={template})",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
