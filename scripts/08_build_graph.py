"""
Step 08 — NetworkX 그래프 빌드 (계층형).

노드:
  - 필드 노드 (854) — 메타: cluster, category, alpha_count, is_centroid
  - 클러스터 노드 (가상, 'C_k' 이름) — 센트로이드 필드를 포함

엣지:
  - inter-field "top-K nearest" — 각 필드의 K 개 이웃만 연결 (K=config)
  - inter-field "top-K farthest" — 다양화 후보를 시각적으로 표시
  - intra-cluster — 센트로이드 ↔ 멤버 (연결 구조 표현)

출력:
  output/field_graph.gpickle  (pickle of networkx.Graph)
  output/field_graph_edges.csv  분석용
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
)

STEP = "08_graph"


def main():
    config = load_config()
    log_line("08", "start")

    data_dir = resolve_path(config["paths"]["data_dir"])
    sim_path = data_dir / "similarity_combined.npy"
    sim_ids_path = data_dir / "similarity_field_ids.json"
    clusters_path = data_dir / "clusters_behavior.json"
    corpus_path = resolve_path(config["paths"]["corpus_file"])
    meta_path = data_dir / "single_field_meta.jsonl"
    out_pickle = resolve_path(config["paths"]["graph_pickle"])
    out_csv = out_pickle.parent / "field_graph_edges.csv"
    ensure_parent(out_pickle)

    import networkx as nx

    combined = np.load(sim_path).astype(np.float32)
    field_ids = json.loads(sim_ids_path.read_text(encoding="utf-8"))
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    assignments = clusters["assignments"]
    cluster_map: dict[str, list[str]] = clusters["clusters"]
    centroids: dict[str, str] = clusters.get("centroids", {})

    # 필드 메타 로드
    field_meta: dict[str, dict] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                field_meta[r["field_id"]] = r

    # 단일 시뮬 결과 메타
    sim_meta: dict[str, dict] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sim_meta[r["field_id"]] = r

    N = len(field_ids)
    idx_of = {fid: i for i, fid in enumerate(field_ids)}

    G = nx.Graph()

    # 센트로이드 셋
    centroid_set = set(centroids.values())

    # 노드 추가
    for fid in field_ids:
        cid = str(assignments.get(fid, -1))
        meta = field_meta.get(fid, {})
        sm = sim_meta.get(fid, {})
        G.add_node(
            fid,
            kind="field",
            cluster=cid,
            is_centroid=(fid in centroid_set),
            category=meta.get("category_name", ""),
            subcategory=meta.get("subcategory_name", ""),
            dataset=meta.get("dataset_name", ""),
            alpha_count=int(meta.get("alpha_count", 0)),
            coverage=float(meta.get("coverage", 0.0)),
            sharpe=float(sm.get("sharpe", 0.0)),
            fitness=float(sm.get("fitness", 0.0)),
            turnover=float(sm.get("turnover", 0.0)),
        )

    # top-K 이웃 엣지 (유사도 높은 것)
    K_near = int(config.get("graph", {}).get("top_k_near", 8))
    K_far = int(config.get("graph", {}).get("top_k_far", 3))

    # 대각 무시
    sim_copy = combined.copy()
    np.fill_diagonal(sim_copy, -np.inf)
    # top-K 가까운 이웃 인덱스 (각 행)
    near_idx = np.argsort(-sim_copy, axis=1)[:, :K_near]

    # top-K 먼 이웃 (다양화 후보)
    sim_copy2 = combined.copy()
    np.fill_diagonal(sim_copy2, np.inf)
    far_idx = np.argsort(sim_copy2, axis=1)[:, :K_far]

    added_near = set()
    added_far = set()
    for i in range(N):
        for j in near_idx[i]:
            if i == j:
                continue
            key = tuple(sorted([i, int(j)]))
            if key in added_near:
                continue
            added_near.add(key)
            sim = float(combined[i, j])
            fu, fv = field_ids[i], field_ids[j]
            G.add_edge(
                fu, fv,
                kind="near",
                weight=sim,
                similarity=sim,
                distance=1.0 - sim,
            )
        for j in far_idx[i]:
            if i == j:
                continue
            key = tuple(sorted([i, int(j)]))
            if key in added_far:
                continue
            added_far.add(key)
            sim = float(combined[i, j])
            fu, fv = field_ids[i], field_ids[j]
            if G.has_edge(fu, fv):
                continue  # 이미 near 로 들어감 (드문 일)
            G.add_edge(
                fu, fv,
                kind="far",
                weight=1.0 - sim,
                similarity=sim,
                distance=1.0 - sim,
            )

    log_line("08", f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
                   f"near={len(added_near)}, far={len(added_far)}")

    # pickle
    with open(out_pickle, "wb") as f:
        pickle.dump(G, f)

    # CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("source,target,kind,weight,similarity,distance\n")
        for u, v, d in G.edges(data=True):
            f.write(
                f"{u},{v},{d.get('kind','')},{d.get('weight','')},"
                f"{d.get('similarity','')},{d.get('distance','')}\n"
            )

    log_line("08", f"saved → {out_pickle}")
    update_step(config, STEP, progress_value=0.75,
                message=f"08 graph · nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
