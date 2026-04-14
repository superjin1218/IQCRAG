"""
Step 01 — 필드 코퍼스 빌드.

wq_brain_all_fields.csv 를 읽고, usable_fields.json 과 coverage/alphaCount 기준으로
필터링한 뒤, 필드별로 임베딩에 넣을 텍스트 한 줄을 만든다.

출력:
  data/field_corpus.jsonl  (한 줄당 하나의 JSON 오브젝트)
    {
      "field_id": "anl4_afv4_eps_mean",
      "text": "anl4_afv4_eps_mean | Earnings per share - mean of estimations ... | dataset=Analyst Estimate Data ... | category=Analyst/Analyst Estimates",
      "description": "...",
      "dataset_name": "...",
      "category_name": "...",
      "subcategory_name": "...",
      "coverage": 1.0,
      "alpha_count": 156,
      "user_count": 84,
      "type": "MATRIX"
    }
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    FIELD_GRAPH_ROOT,
    load_config,
    resolve_path,
    ensure_parent,
    update_step,
    log_line,
)

STEP = "01_corpus"


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="읽는 중")
    log_line(STEP, "start")

    csv_path = resolve_path(config["input"]["all_fields_csv"])
    usable_path = resolve_path(config["input"]["usable_fields_json"])
    out_path = resolve_path(config["paths"]["corpus_file"])
    ensure_parent(out_path)

    # usable 필드 화이트리스트
    # usable_fields.json 스키마:
    #   { "banned": [...], "total_usable": int, "by_category": { "Cat1": [..], "Cat2": [..] } }
    usable_set = None
    banned_set: set = set()
    if config["filter"].get("use_usable_fields_only", False) and usable_path.exists():
        try:
            usable_data = json.loads(usable_path.read_text(encoding="utf-8"))
            if isinstance(usable_data, dict):
                if "by_category" in usable_data and isinstance(usable_data["by_category"], dict):
                    usable_set = set()
                    for cat, fields in usable_data["by_category"].items():
                        if isinstance(fields, list):
                            usable_set.update(fields)
                    banned_set = set(usable_data.get("banned", []) or [])
                elif "fields" in usable_data:
                    usable_set = set(usable_data["fields"])
                else:
                    usable_set = set()
                    for v in usable_data.values():
                        if isinstance(v, list):
                            usable_set.update(v)
            elif isinstance(usable_data, list):
                usable_set = set(usable_data)
            log_line(STEP, f"usable 화이트리스트 {len(usable_set) if usable_set else 0} 개, banned {len(banned_set)} 개")
        except Exception as e:
            log_line(STEP, f"usable_fields 로드 실패, 전체 사용: {e}")
            usable_set = None

    min_cov = float(config["filter"].get("min_coverage", 0.0))
    min_ac = int(config["filter"].get("min_alpha_count", 0))

    kept = 0
    dropped = 0
    rows_out = []
    seen_fids: set = set()  # CSV 에 중복 행이 있을 수 있어 dedup

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["field_id"].strip()
            if fid in seen_fids:
                # CSV 중복 행 → 첫 번째만 유지
                continue
            if usable_set is not None and fid not in usable_set:
                dropped += 1
                continue
            if fid in banned_set:
                dropped += 1
                continue
            seen_fids.add(fid)
            try:
                cov = float(row.get("coverage", 0) or 0)
            except ValueError:
                cov = 0.0
            try:
                ac = int(float(row.get("alphaCount", 0) or 0))
            except ValueError:
                ac = 0
            if cov < min_cov or ac < min_ac:
                dropped += 1
                continue

            desc = row.get("description", "").strip()
            ds_name = row.get("dataset_name", "").strip()
            cat_name = row.get("category_name", "").strip()
            sub_name = row.get("subcategory_name", "").strip()

            text = f"{fid} | {desc} | dataset={ds_name} | category={cat_name}/{sub_name}"
            rows_out.append({
                "field_id": fid,
                "text": text,
                "description": desc,
                "dataset_name": ds_name,
                "category_name": cat_name,
                "subcategory_name": sub_name,
                "coverage": cov,
                "alpha_count": ac,
                "user_count": int(float(row.get("userCount", 0) or 0)),
                "type": row.get("type", "").strip(),
            })
            kept += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log_line(STEP, f"kept={kept}, dropped={dropped} → {out_path}")
    update_step(
        config, STEP,
        status="done", progress_value=1.0,
        message=f"{kept} 필드 kept, {dropped} dropped",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
