"""
Step 10 — 다양화 후보 추천 CLI.

입력 알파 (수식 or 필드 리스트) 를 받아서 combined similarity 상 거리가 먼
필드를 top-N 추천한다.

사용 예:
  python scripts/10_find_diversifier.py --alpha-expr "add(rank(ts_backfill(fnd6_fopo, 20)), rank(ts_backfill(eps, 20)))"
  python scripts/10_find_diversifier.py --fields fnd6_fopo,eps --top 15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_path

FIELD_ID_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def main():
    parser = argparse.ArgumentParser(description="Field diversifier recommender")
    parser.add_argument("--alpha-expr", type=str, default=None)
    parser.add_argument("--fields", type=str, default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--show-similar", action="store_true",
                        help="가까운 필드(redundant) 도 함께 표시")
    args = parser.parse_args()

    if not args.alpha_expr and not args.fields:
        parser.error("--alpha-expr 또는 --fields 필요")

    config = load_config()
    data_dir = resolve_path(config["paths"]["data_dir"])

    sim = np.load(data_dir / "similarity_combined.npy").astype(np.float32)
    sim_pnl = np.load(data_dir / "similarity_pnl.npy").astype(np.float32)
    sim_text = np.load(data_dir / "similarity_text.npy").astype(np.float32)
    field_ids = json.loads((data_dir / "similarity_field_ids.json").read_text(encoding="utf-8"))
    idx_of = {fid: i for i, fid in enumerate(field_ids)}
    known = set(field_ids)

    # 입력 파싱
    if args.fields:
        input_fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    else:
        tokens = set(FIELD_ID_RE.findall(args.alpha_expr))
        input_fields = sorted(tokens & known)

    if not input_fields:
        print("입력 알파에서 알려진 필드 찾지 못함")
        return

    print(f"입력 필드: {', '.join(input_fields)}")

    valid_inputs = [f for f in input_fields if f in idx_of]
    if not valid_inputs:
        print("알려진 필드가 없음 (시뮬 안 된 필드일 수 있음)")
        return

    # 입력 필드들의 평균 similarity row → 각 후보 필드의 average similarity to input set
    in_idxs = [idx_of[f] for f in valid_inputs]
    avg_sim = sim[in_idxs, :].mean(axis=0)            # (N,) 입력과의 평균 유사도
    avg_pnl = sim_pnl[in_idxs, :].mean(axis=0)
    avg_text = sim_text[in_idxs, :].mean(axis=0)

    # 후보: 입력 필드 제외
    mask = np.ones(len(field_ids), dtype=bool)
    for i in in_idxs:
        mask[i] = False

    # 다양화 = 낮은 유사도
    cand_idx = np.where(mask)[0]
    cand_avg = avg_sim[cand_idx]
    order_far = cand_idx[np.argsort(cand_avg)]        # 오름차순 (먼 순)
    order_near = cand_idx[np.argsort(-cand_avg)]      # 내림차순 (가까운 순)

    print()
    print(f"=== DIVERSIFIER TOP {args.top} (입력과 가장 먼 필드) ===")
    print(f"{'#':<4}{'field_id':<40}{'sim':>7}{'pnl':>7}{'text':>7}")
    for k, j in enumerate(order_far[:args.top], 1):
        fid = field_ids[j]
        print(f"{k:<4}{fid:<40}{avg_sim[j]:>7.3f}{avg_pnl[j]:>7.3f}{avg_text[j]:>7.3f}")

    if args.show_similar:
        print()
        print(f"=== REDUNDANT TOP {args.top} (입력과 가장 가까운 필드) ===")
        print(f"{'#':<4}{'field_id':<40}{'sim':>7}{'pnl':>7}{'text':>7}")
        for k, j in enumerate(order_near[:args.top], 1):
            fid = field_ids[j]
            print(f"{k:<4}{fid:<40}{avg_sim[j]:>7.3f}{avg_pnl[j]:>7.3f}{avg_text[j]:>7.3f}")


if __name__ == "__main__":
    main()
