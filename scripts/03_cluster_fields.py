"""
Step 03 — 필드 클러스터링 (HDBSCAN).

embeddings.npy 를 읽어서 HDBSCAN 으로 클러스터링한다. HDBSCAN 은 클러스터 수를
직접 지정하지 않고 min_cluster_size 로 제어하므로, target_clusters (기본 100) 에
가까워지도록 min_cluster_size 를 자동 조정한다.

출력:
  data/clusters.json
    {
      "algorithm": "hdbscan",
      "n_clusters": 102,
      "n_noise": 37,
      "params": {"min_cluster_size": 14, ...},
      "assignments": {"field_id": cluster_label, ...},  # -1 = noise
      "clusters": {
        "0": ["field_a", "field_b", ...],
        "1": [...]
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

STEP = "03_cluster"


def _run_hdbscan(vecs: np.ndarray, min_cluster_size: int, min_samples: int, epsilon: float, metric: str):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric=metric,
    )
    labels = clusterer.fit_predict(vecs)
    return labels


def _count_clusters(labels: np.ndarray) -> int:
    s = set(labels.tolist())
    s.discard(-1)
    return len(s)


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    emb_path = resolve_path(config["paths"]["embeddings_file"])
    ids_path = resolve_path(config["paths"]["field_ids_file"])
    out_path = resolve_path(config["paths"]["clusters_file"])
    ensure_parent(out_path)

    if not emb_path.exists():
        raise RuntimeError("embeddings 가 없습니다. 02 단계를 먼저 실행하세요.")

    vecs = np.load(emb_path).astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    N = len(ids)
    log_line(STEP, f"vecs shape={vecs.shape}, ids={N}")

    target = int(config["cluster"].get("target_clusters", 100))
    base_mcs = int(config["cluster"].get("min_cluster_size", 5))
    min_samples = int(config["cluster"].get("min_samples", 2))
    epsilon = float(config["cluster"].get("cluster_selection_epsilon", 0.0))
    metric = config["cluster"].get("metric", "euclidean")

    # min_cluster_size 를 자동 탐색: 작은 값부터 키워가며 target 에 가까운 n_clusters 찾기
    # 노이즈 최소화를 위해 작은 값 후보부터 넉넉하게 포함한다.
    best_labels = None
    best_mcs = base_mcs
    best_diff = float("inf")
    best_score = float("inf")  # diff + noise_penalty
    tried = []
    base_candidates = {2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25}
    # N/target 비례 값도 추가
    base_candidates.update({
        max(2, int(N / target * x)) for x in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5)
    })
    candidates = sorted(base_candidates)

    # 점수 = |n_clusters - target| + 0.5 * (noise_count / N * target)
    # → target 근접 + 노이즈 적을수록 좋음
    for i, mcs in enumerate(candidates):
        log_line(STEP, f"try min_cluster_size={mcs}")
        labels = _run_hdbscan(vecs, mcs, min_samples, epsilon, metric)
        nc = _count_clusters(labels)
        noise = int((labels == -1).sum())
        tried.append({"min_cluster_size": mcs, "n_clusters": nc, "n_noise": noise})
        diff = abs(nc - target)
        score = diff + 0.5 * (noise / N * target)
        if score < best_score:
            best_score = score
            best_diff = diff
            best_labels = labels
            best_mcs = mcs
        update_step(
            config, STEP, progress_value=(i + 1) / max(1, len(candidates)),
            message=f"try mcs={mcs} → {nc} clusters, noise={noise}",
        )

    if best_labels is None:
        raise RuntimeError("HDBSCAN 결과가 없습니다.")

    labels = best_labels
    n_clusters = _count_clusters(labels)
    n_noise = int((labels == -1).sum())
    log_line(STEP, f"best min_cluster_size={best_mcs}, n_clusters={n_clusters}, n_noise={n_noise}")

    # 저장
    assignments = {ids[i]: int(labels[i]) for i in range(N)}
    clusters_map: dict[str, list[str]] = {}
    for fid, lab in assignments.items():
        clusters_map.setdefault(str(lab), []).append(fid)

    payload = {
        "algorithm": "hdbscan",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "target_clusters": target,
        "params": {
            "min_cluster_size": best_mcs,
            "min_samples": min_samples,
            "cluster_selection_epsilon": epsilon,
            "metric": metric,
        },
        "tried": tried,
        "assignments": assignments,
        "clusters": clusters_map,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(STEP, f"저장 → {out_path}")

    update_step(
        config, STEP,
        status="done", progress_value=1.0,
        message=f"{n_clusters} clusters, {n_noise} noise (target={target})",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
