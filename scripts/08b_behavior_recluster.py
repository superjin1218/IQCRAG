"""
Step 08b — 행동 기반 재클러스터링.

similarity_combined 를 distance 로 변환해서 HDBSCAN 돌림.
텍스트 기반 클러스터 (clusters.json) 와 비교해 얼마나 다른지도 리포트.

출력:
  data/clusters_behavior.json
  data/cluster_comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, update_step, log_line,
)

STEP = "08_graph"


def main():
    config = load_config()
    log_line("08b", "start")

    data_dir = resolve_path(config["paths"]["data_dir"])
    sim_path = data_dir / "similarity_combined.npy"
    sim_ids_path = data_dir / "similarity_field_ids.json"
    text_clusters_path = resolve_path(config["paths"]["clusters_file"])

    combined = np.load(sim_path).astype(np.float64)
    field_ids = json.loads(sim_ids_path.read_text(encoding="utf-8"))
    N = len(field_ids)

    # distance = 1 - similarity (대각은 0)
    distance = 1.0 - combined
    np.fill_diagonal(distance, 0.0)
    # 혹시 미세 음수 생기면 클립
    distance = np.clip(distance, 0.0, None)
    # 대칭 강제
    distance = (distance + distance.T) / 2.0

    import hdbscan

    target = int(config["cluster"].get("target_clusters", 100))
    # N/target 주변으로 탐색
    candidates = sorted({
        max(2, int(N / target * x))
        for x in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    })
    candidates = sorted(set(candidates) | {3, 4, 5, 8})

    best_labels = None
    best_mcs = None
    best_score = float("inf")
    tried = []
    for mcs in candidates:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=2,
            metric="precomputed",
            cluster_selection_epsilon=0.0,
        )
        labels = clusterer.fit_predict(distance)
        n_cl = len(set(labels.tolist()) - {-1})
        noise = int((labels == -1).sum())
        diff = abs(n_cl - target)
        score = diff + 0.5 * (noise / N * target)
        tried.append({"mcs": mcs, "n_clusters": n_cl, "noise": noise, "score": score})
        if score < best_score:
            best_score = score
            best_labels = labels
            best_mcs = mcs

    labels = best_labels
    n_cl = len(set(labels.tolist()) - {-1})
    noise = int((labels == -1).sum())
    log_line("08b", f"behavior clusters: {n_cl}, noise: {noise}, mcs={best_mcs}")

    # 저장
    assignments = {field_ids[i]: int(labels[i]) for i in range(N)}
    clusters_map: dict[str, list[str]] = {}
    for fid, lab in assignments.items():
        clusters_map.setdefault(str(lab), []).append(fid)

    # centroid 대표 필드 찾기 (클러스터 내부에서 다른 멤버들과 가장 상관 높은 필드)
    centroids: dict[str, str] = {}
    for cid, members in clusters_map.items():
        if cid == "-1" or not members:
            continue
        if len(members) == 1:
            centroids[cid] = members[0]
            continue
        idxs = [field_ids.index(f) for f in members]
        sub = combined[np.ix_(idxs, idxs)]
        avg_sim = sub.mean(axis=1)  # 각 멤버의 클러스터 내부 평균 유사도
        best_local = int(np.argmax(avg_sim))
        centroids[cid] = members[best_local]

    out = {
        "algorithm": "hdbscan_precomputed",
        "n_clusters": n_cl,
        "n_noise": noise,
        "best_mcs": best_mcs,
        "tried": tried,
        "assignments": assignments,
        "clusters": clusters_map,
        "centroids": centroids,
    }
    (data_dir / "clusters_behavior.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 텍스트 클러스터와 비교
    try:
        text_clusters = json.loads(text_clusters_path.read_text(encoding="utf-8"))
        text_assign = text_clusters.get("assignments", {})

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        common_fids = [f for f in field_ids if f in text_assign]
        labels_text = [text_assign[f] for f in common_fids]
        labels_beh = [assignments[f] for f in common_fids]
        ari = adjusted_rand_score(labels_text, labels_beh)
        nmi = normalized_mutual_info_score(labels_text, labels_beh)
        comparison = {
            "n_common": len(common_fids),
            "adjusted_rand_index": float(ari),
            "normalized_mutual_info": float(nmi),
            "text_n_clusters": len(set(labels_text) - {-1}),
            "behavior_n_clusters": n_cl,
        }
    except Exception as e:
        comparison = {"error": str(e)[:200]}

    (data_dir / "cluster_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_line("08b", f"comparison: {comparison}")
    update_step(config, STEP, progress_value=0.55,
                message=f"08b done · behavior clusters={n_cl}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
