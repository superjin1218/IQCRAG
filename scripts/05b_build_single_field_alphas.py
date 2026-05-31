"""
Step 05b — 단일 필드 probe alpha 수식 생성 (v2: CSV 타입 디스패치 + 멀티계정 사전할당).

입력:
  config.probe.field_list_file (CSV) — 필수 컬럼: field_id, type, description
  config.probe.templates         — type 별 수식 템플릿
  config.probe.backfill_window   — ts_backfill 윈도우 (252)
  config.accounts.list           — 계정 + max_concurrent (a1×3 + a2×3 = 6 슬롯)

처리 규칙:
  type ∈ skip_types (GROUP/UNIVERSE/SYMBOL) → skip "type_blacklist:{t}"
  type == "VECTOR":
    suffix ∈ vector_skip_suffixes        → skip "categorical_suffix:{sfx}"
    description 에 keyword 포함          → skip "categorical_desc:{kw}"
    그 외                                → VECTOR_NUMERIC 템플릿 (vec_avg)
  type == "MATRIX"                       → MATRIX 템플릿 (plain)
  그 외                                  → skip "unknown_type"

각 eligible 필드를 sorted(field_id) 순으로 (account, shard) 슬롯에 라운드로빈.

출력:
  data/probe_alphas.jsonl
    {"field_id","type","reducer","expr","backfill_window","account_id","shard_idx","description"}
  data/probe_alphas_skipped.jsonl
    {"field_id","type","skip_reason","description"}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
)

STEP = "05_probe_alphas"


def load_field_catalog(csv_path: Path) -> list[dict]:
    """field list CSV 로드. 필수 컬럼: field_id, type, description."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"필드 리스트 CSV 가 없습니다: {csv_path}\n"
            f"사용자가 references/fields_v2.csv 를 제공해야 합니다.\n"
            f"필수 컬럼: field_id, type, description"
        )
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fid = (r.get("field_id") or "").strip()
            if not fid:
                continue
            rows.append({
                "field_id": fid,
                "type": (r.get("type") or "").strip().upper(),
                "description": (r.get("description") or "").strip(),
            })
    return rows


def build_expr(field_id: str, ftype: str, description: str, cfg: dict) -> tuple[str, str, str]:
    """returns (status, expr_or_reason, reducer)
    status ∈ {"ok", "skip"}
    """
    probe = cfg["probe"]
    bw = int(probe["backfill_window"])
    templates = probe["templates"]
    skip_types = set(probe.get("skip_types", []))
    skip_suffixes = list(probe.get("vector_skip_suffixes", []))
    skip_keywords = [k.lower() for k in probe.get("vector_skip_keywords", [])]

    if ftype in skip_types:
        return "skip", f"type_blacklist:{ftype}", "none"

    if ftype == "VECTOR":
        for sfx in skip_suffixes:
            if field_id.endswith(sfx):
                return "skip", f"categorical_suffix:{sfx}", "none"
        desc_l = description.lower()
        for kw in skip_keywords:
            if kw and kw in desc_l:
                return "skip", f"categorical_desc:{kw}", "none"
        expr = templates["VECTOR_NUMERIC"].format(fid=field_id, bw=bw)
        return "ok", expr, "vec_avg"

    if ftype == "MATRIX":
        expr = templates["MATRIX"].format(fid=field_id, bw=bw)
        return "ok", expr, "none"

    return "skip", f"unknown_type:{ftype or 'EMPTY'}", "none"


def assign_account_shard(field_ids: list[str], accounts_cfg: dict) -> dict[str, tuple[str, int]]:
    """sorted(field_ids) 순서대로 (account, shard) 라운드로빈 할당."""
    accounts = accounts_cfg.get("list", []) or []
    if not accounts:
        raise ValueError("config.accounts.list 가 비어있습니다.")
    default_n = int(accounts_cfg.get("default_max_concurrent", 3))
    per_acct = [int(a.get("max_concurrent", default_n)) for a in accounts]
    ids = [a["id"] for a in accounts]

    ordered = sorted(field_ids)
    assignment: dict[str, tuple[str, int]] = {}
    for i, fid in enumerate(ordered):
        acct_idx = i % len(ids)
        acct = ids[acct_idx]
        # 같은 계정 내에서 다음 슬롯으로 분배
        shard = (i // len(ids)) % per_acct[acct_idx]
        assignment[fid] = (acct, shard)
    return assignment


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    field_csv = resolve_path(config["probe"]["field_list_file"])
    log_line(STEP, f"필드 CSV: {field_csv}")
    catalog = load_field_catalog(field_csv)
    log_line(STEP, f"CSV 로드: {len(catalog)} 필드")

    # 1차: 분기 → eligible / skipped
    eligible: list[dict] = []
    skipped: list[dict] = []
    for r in catalog:
        status, payload, reducer = build_expr(
            r["field_id"], r["type"], r["description"], config
        )
        if status == "ok":
            eligible.append({
                "field_id": r["field_id"],
                "type": r["type"],
                "reducer": reducer,
                "expr": payload,
                "backfill_window": int(config["probe"]["backfill_window"]),
                "description": r["description"],
            })
        else:
            skipped.append({
                "field_id": r["field_id"],
                "type": r["type"],
                "skip_reason": payload,
                "description": r["description"],
            })

    # 2차: account/shard 사전 할당
    eligible_ids = [e["field_id"] for e in eligible]
    accounts_cfg = config.get("accounts", {})
    if accounts_cfg.get("list"):
        assignment = assign_account_shard(eligible_ids, accounts_cfg)
        for e in eligible:
            acct, shard = assignment[e["field_id"]]
            e["account_id"] = acct
            e["shard_idx"] = shard
    else:
        # 레거시 모드: account/shard 비움
        for e in eligible:
            e["account_id"] = None
            e["shard_idx"] = None

    # 출력
    out_path = resolve_path(config["paths"]["probe_alphas_file"])
    skip_path = out_path.parent / "probe_alphas_skipped.jsonl"
    ensure_parent(out_path)

    eligible_sorted = sorted(eligible, key=lambda x: x["field_id"])
    with open(out_path, "w", encoding="utf-8") as f:
        for row in eligible_sorted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(skip_path, "w", encoding="utf-8") as f:
        for row in skipped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 분포 요약 (skip_reason 별)
    from collections import Counter
    skip_counter = Counter(s["skip_reason"].split(":")[0] for s in skipped)
    type_counter = Counter(e["type"] for e in eligible)
    if accounts_cfg.get("list"):
        slot_counter = Counter((e["account_id"], e["shard_idx"]) for e in eligible)
        slot_msg = " ".join(f"{a}/s{s}={n}" for (a, s), n in sorted(slot_counter.items()))
    else:
        slot_msg = "(single-account legacy)"

    log_line(STEP, f"eligible: {len(eligible)} ({dict(type_counter)})")
    log_line(STEP, f"skipped : {len(skipped)} ({dict(skip_counter)})")
    log_line(STEP, f"slots   : {slot_msg}")
    log_line(STEP, f"→ {out_path}")
    log_line(STEP, f"→ {skip_path}")

    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"eligible={len(eligible)} skipped={len(skipped)} bw={config['probe']['backfill_window']}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
