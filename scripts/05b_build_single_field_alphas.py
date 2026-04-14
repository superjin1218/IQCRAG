"""
Step 05b — 단일 필드 probe alpha 수식 생성.

corpus 의 854 필드마다 probe 알파 1개씩:
    rank(ts_backfill(X, 20))                 (MATRIX 기본)
    rank(vec_avg(ts_backfill(X, 20)))        (snt_ / scl12_ / nws12_ 접두사만)

출력:
  data/single_field_probes.jsonl
    {"field_id": "...", "expr": "rank(ts_backfill(...))", "wrap": "plain"|"vec_avg"}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
)

STEP = "05_probe_alphas"

VEC_PREFIXES = ("snt_", "scl12_", "nws12_")


def build_expr(field_id: str) -> tuple[str, str]:
    if any(field_id.startswith(p) for p in VEC_PREFIXES):
        return f"rank(vec_avg(ts_backfill({field_id}, 20)))", "vec_avg"
    return f"rank(ts_backfill({field_id}, 20))", "plain"


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    corpus_path = resolve_path(config["paths"]["corpus_file"])
    out_path = resolve_path(config["paths"]["probe_alphas_file"])
    ensure_parent(out_path)

    rows = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            fid = r["field_id"]
            expr, wrap = build_expr(fid)
            rows.append({"field_id": fid, "expr": expr, "wrap": wrap})

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_line(STEP, f"단일 필드 probe {len(rows)} 개 생성 → {out_path}")
    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"{len(rows)} 단일 필드 알파 수식 생성",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
