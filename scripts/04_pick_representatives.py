"""
Step 04 — 클러스터 대표 필드 선정 (하이브리드).

각 클러스터마다 score = 0.7 * centroid_cosine + 0.3 * log_alphacount_normalized 가
가장 높은 필드를 대표로 선정한다. 노이즈(-1) 클러스터는 대표를 뽑지 않는다.

출력:
  data/representatives.json
    {
      "n_clusters": 100,
      "weights": {"centroid": 0.7, "alphacount": 0.3},
      "representatives": {
        "0": {"field_id": "fnd6_fopo", "score": 0.93, "centroid_sim": 0.98, "log_ac_norm": 0.81, "cluster_size": 14},
        ...
      }
    }
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

STEP = "04_representatives"


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    emb_path = resolve_path(config["paths"]["embeddings_file"])
    ids_path = resolve_path(config["paths"]["field_ids_file"])
    corpus_path = resolve_path(config["paths"]["corpus_file"])
    clusters_path = resolve_path(config["paths"]["clusters_file"])
    out_path = resolve_path(config["paths"]["representatives_file"])
    ensure_parent(out_path)

    vecs = np.load(emb_path).astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    id2idx = {fid: i for i, fid in enumerate(ids)}

    # alphaCount 맵 (corpus 에서 읽음)
    ac_map: dict[str, int] = {}
    meta_map: dict[str, dict] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ac_map[r["field_id"]] = int(r.get("alpha_count", 0))
            meta_map[r["field_id"]] = {
                "dataset": r.get("dataset_name", ""),
                "category": r.get("category_name", ""),
                "subcategory": r.get("subcategory_name", ""),
            }

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    cluster_map: dict[str, list[str]] = clusters["clusters"]

    w_c = float(config["representative"]["centroid_weight"])
    w_a = float(config["representative"]["alphacount_weight"])
    tie_break = config["representative"].get("tie_break", "alphacount")

    # 전체 alphaCount 로그 정규화
    all_acs = np.array(list(ac_map.values()), dtype=np.float32)
    log_max = float(np.log1p(all_acs.max()) or 1.0)

    representatives: dict[str, dict] = {}
    total_clusters = sum(1 for k in cluster_map.keys() if k != "-1")
    processed = 0

    for cid, fields in cluster_map.items():
        if cid == "-1":
            # 노이즈는 대표 안 뽑음
            continue
        if not fields:
            continue

        # 클러스터 내부 벡터
        cluster_idxs = [id2idx[f] for f in fields if f in id2idx]
        if not cluster_idxs:
            continue
        cluster_vecs = vecs[cluster_idxs]  # (k, D)
        centroid = cluster_vecs.mean(axis=0)
        # 코사인 유사도 (이미 정규화 가정, 안전하게 재정규화)
        cn = centroid / (np.linalg.norm(centroid) + 1e-12)
        vn = cluster_vecs / (np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-12)
        sims = vn @ cn  # (k,)

        # alphaCount 정규화
        acs = np.array([ac_map.get(f, 0) for f in fields if f in id2idx], dtype=np.float32)
        log_acs = np.log1p(acs) / log_max  # [0, 1]

        # 스코어
        scores = w_c * sims + w_a * log_acs

        # 최고 점수, 동점이면 tie_break
        best_local = int(scores.argmax())
        if (scores == scores.max()).sum() > 1:
            ties = np.where(scores == scores.max())[0]
            if tie_break == "alphacount":
                best_local = int(ties[np.argmax(acs[ties])])
            else:
                best_local = int(ties[0])

        valid_fields = [f for f in fields if f in id2idx]
        chosen = valid_fields[best_local]
        representatives[cid] = {
            "field_id": chosen,
            "score": float(scores[best_local]),
            "centroid_sim": float(sims[best_local]),
            "log_ac_norm": float(log_acs[best_local]),
            "alpha_count": int(ac_map.get(chosen, 0)),
            "cluster_size": len(valid_fields),
            "meta": meta_map.get(chosen, {}),
        }

        processed += 1
        if processed % 10 == 0:
            update_step(config, STEP, progress_value=processed / total_clusters,
                        message=f"{processed}/{total_clusters}")

    out_path.write_text(
        json.dumps({
            "n_clusters": len(representatives),
            "weights": {"centroid": w_c, "alphacount": w_a},
            "tie_break": tie_break,
            "representatives": representatives,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_line(STEP, f"대표 {len(representatives)}개 선정 → {out_path}")
    update_step(
        config, STEP,
        status="done", progress_value=1.0,
        message=f"{len(representatives)} 대표 필드 선정",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
