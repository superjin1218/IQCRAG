"""
Step 08a — 유사도 매트릭스 빌드.

단계:
  1. pnl_records.jsonl → (N, T) wide 매트릭스
  2. cross-sectional demean (각 날짜의 알파 평균 제거)
  3. np.corrcoef → (N, N) PnL 상관
  4. embeddings.npy + field_ids.json 으로 (N, N) text cosine 계산
  5. combined_sim = α × |pnl_corr| + (1-α) × text_cos
  6. 저장: pnl_corr, text_cos, combined_sim 모두 (다뷰 시각화에 필요)

출력:
  data/similarity_pnl.npy         (N, N) float32
  data/similarity_text.npy        (N, N) float32
  data/similarity_combined.npy    (N, N) float32
  data/similarity_field_ids.json  공통 인덱스 field_id 순서
  data/similarity_meta.json       빌드 설정/통계
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
)

STEP = "08_graph"  # 08 전체 하나의 스텝으로 묶음 (대시보드 단순화)


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="08a similarity")
    log_line("08a", "start")

    data_dir = resolve_path(config["paths"]["data_dir"])
    pnl_path = data_dir / "pnl_records.jsonl"
    meta_path = data_dir / "single_field_meta.jsonl"
    emb_path = resolve_path(config["paths"]["embeddings_file"])
    ids_path = resolve_path(config["paths"]["field_ids_file"])

    if not pnl_path.exists():
        raise RuntimeError(f"{pnl_path} 없음. 07b 먼저 실행.")

    # 1. pnl_records 를 wide 매트릭스로
    log_line("08a", "loading pnl records")
    from collections import defaultdict
    pnl_by_field: dict[str, dict[str, float]] = defaultdict(dict)
    all_dates: set[str] = set()
    with open(pnl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            fid = r["field_id"]
            dt = r["date"]
            pnl_by_field[fid][dt] = float(r["pnl"])
            all_dates.add(dt)

    all_dates_sorted = sorted(all_dates)
    field_ids_pnl = sorted(pnl_by_field.keys())
    T = len(all_dates_sorted)
    N_pnl = len(field_ids_pnl)
    log_line("08a", f"pnl fields={N_pnl}, dates={T}")

    if N_pnl == 0:
        raise RuntimeError("pnl_records.jsonl 이 비어 있음. 시뮬이 완료됐는지 확인.")

    # 공통 date 만 사용 (일부 필드가 일부 날짜 없을 수 있음)
    pnl_mat = np.full((N_pnl, T), np.nan, dtype=np.float32)
    date_idx = {d: i for i, d in enumerate(all_dates_sorted)}
    for i, fid in enumerate(field_ids_pnl):
        for dt, v in pnl_by_field[fid].items():
            pnl_mat[i, date_idx[dt]] = v

    # NaN 이 많은 열 드롭
    valid_cols = ~np.any(np.isnan(pnl_mat), axis=0)
    pnl_mat_v = pnl_mat[:, valid_cols]
    log_line("08a", f"valid dates after nan drop: {pnl_mat_v.shape[1]}")

    # 2. cross-sectional demean
    demean = bool(config.get("similarity", {}).get("pnl_demean", True))
    if demean:
        daily_mean = pnl_mat_v.mean(axis=0, keepdims=True)
        pnl_idio = pnl_mat_v - daily_mean
        log_line("08a", "cross-sectional demean applied")
    else:
        pnl_idio = pnl_mat_v

    # 3. PnL 상관
    pnl_corr = np.corrcoef(pnl_idio).astype(np.float32)
    pnl_corr_abs = np.abs(pnl_corr)
    log_line("08a", f"pnl corr matrix: {pnl_corr.shape}")

    # 4. text cosine: 공통 field_id 만 (pnl 에 있는 필드)
    embeddings = np.load(emb_path).astype(np.float32)
    all_field_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    fid_to_idx = {fid: i for i, fid in enumerate(all_field_ids)}

    # pnl 기준으로 정렬
    keep_mask = [fid in fid_to_idx for fid in field_ids_pnl]
    keep_field_ids = [field_ids_pnl[i] for i, k in enumerate(keep_mask) if k]
    keep_idx_in_pnl = [i for i, k in enumerate(keep_mask) if k]

    pnl_corr_final = pnl_corr[np.ix_(keep_idx_in_pnl, keep_idx_in_pnl)]
    pnl_corr_abs_final = pnl_corr_abs[np.ix_(keep_idx_in_pnl, keep_idx_in_pnl)]

    emb_sub = embeddings[[fid_to_idx[f] for f in keep_field_ids]]
    # 이미 L2 정규화됐다 가정. cosine = dot
    text_cos = emb_sub @ emb_sub.T
    text_cos = np.clip(text_cos, -1.0, 1.0).astype(np.float32)
    log_line("08a", f"text cosine matrix: {text_cos.shape}")

    # 5. combined similarity
    α = float(config.get("similarity", {}).get("pnl_weight", 0.85))
    combined = α * pnl_corr_abs_final + (1.0 - α) * text_cos
    combined = combined.astype(np.float32)
    # 대각 1
    np.fill_diagonal(combined, 1.0)

    # 6. 저장
    np.save(data_dir / "similarity_pnl.npy", pnl_corr_final)
    np.save(data_dir / "similarity_text.npy", text_cos)
    np.save(data_dir / "similarity_combined.npy", combined)

    sim_ids_path = data_dir / "similarity_field_ids.json"
    sim_ids_path.write_text(
        json.dumps(keep_field_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sim_meta = {
        "n_fields": len(keep_field_ids),
        "n_dates_raw": T,
        "n_dates_valid": int(pnl_mat_v.shape[1]),
        "pnl_demean": demean,
        "pnl_weight": α,
        "text_weight": 1.0 - α,
        "normalization": config.get("similarity", {}).get("normalization", "none"),
    }
    (data_dir / "similarity_meta.json").write_text(
        json.dumps(sim_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_line("08a", f"saved. n_fields={len(keep_field_ids)}, α={α}")
    update_step(config, STEP, progress_value=0.33,
                message=f"08a done · {len(keep_field_ids)} fields")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
