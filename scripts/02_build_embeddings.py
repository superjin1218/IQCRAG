"""
Step 02 — 필드 임베딩.

field_corpus.jsonl 을 읽어서 sentence-transformers 모델로 임베딩을 생성하고
numpy 배열로 저장한다. 동시에 ChromaDB 컬렉션에도 넣어서 시맨틱 검색을 가능하게 한다.

출력:
  data/embeddings.npy    — (N, D) 정규화된 벡터
  data/field_ids.json    — 같은 순서의 field_id 리스트
  data/chroma_db/        — ChromaDB 영구 저장소 (컬렉션명: field_graph)
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

STEP = "02_embeddings"


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    corpus_path = resolve_path(config["paths"]["corpus_file"])
    emb_path = resolve_path(config["paths"]["embeddings_file"])
    ids_path = resolve_path(config["paths"]["field_ids_file"])
    chroma_dir = resolve_path(config["paths"]["chroma_dir"])
    ensure_parent(emb_path)
    ensure_parent(ids_path)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 코퍼스 로드
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    N = len(records)
    log_line(STEP, f"코퍼스 {N} 필드 로드")
    if N == 0:
        raise RuntimeError("코퍼스가 비어 있습니다. 01 단계를 먼저 실행하세요.")

    # 모델 로드
    from sentence_transformers import SentenceTransformer
    model_name = config["embedding"]["model"]
    log_line(STEP, f"모델 로드: {model_name}")
    update_step(config, STEP, progress_value=0.05, message=f"모델 로드: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [r["text"] for r in records]
    ids = [r["field_id"] for r in records]
    batch_size = int(config["embedding"].get("batch_size", 32))
    normalize = bool(config["embedding"].get("normalize", True))

    # 배치 임베딩
    log_line(STEP, f"임베딩 생성 시작 (batch={batch_size})")
    all_vecs = []
    total_batches = (N + batch_size - 1) // batch_size
    for b in range(total_batches):
        s = b * batch_size
        e = min(s + batch_size, N)
        batch_texts = texts[s:e]
        vecs = model.encode(
            batch_texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_vecs.append(vecs)
        frac = 0.05 + 0.85 * (b + 1) / total_batches
        update_step(
            config, STEP, progress_value=frac,
            message=f"batch {b+1}/{total_batches}  ({e}/{N})",
        )

    embeddings = np.vstack(all_vecs).astype(np.float32)
    log_line(STEP, f"임베딩 shape={embeddings.shape}")

    # 저장
    np.save(emb_path, embeddings)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    log_line(STEP, f"npy 저장 → {emb_path}")

    # ChromaDB 에 저장
    update_step(config, STEP, progress_value=0.93, message="ChromaDB 저장")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        # 기존 컬렉션이 있으면 삭제 후 재생성
        try:
            client.delete_collection("field_graph")
        except Exception:
            pass
        col = client.create_collection(
            name="field_graph",
            metadata={"model": model_name, "dim": int(embeddings.shape[1])},
        )
        # 메타데이터에 dataset/category/alpha_count 넣어두면 필터 가능
        metas = [
            {
                "dataset": r.get("dataset_name", ""),
                "category": r.get("category_name", ""),
                "subcategory": r.get("subcategory_name", ""),
                "coverage": float(r.get("coverage", 0.0)),
                "alpha_count": int(r.get("alpha_count", 0)),
                "type": r.get("type", ""),
            }
            for r in records
        ]
        col.add(
            ids=ids,
            embeddings=[v.tolist() for v in embeddings],
            documents=texts,
            metadatas=metas,
        )
        log_line(STEP, f"ChromaDB 저장 완료 (items={col.count()})")
    except Exception as e:
        log_line(STEP, f"ChromaDB 저장 실패 (무시): {e}")

    update_step(
        config, STEP,
        status="done", progress_value=1.0,
        message=f"{N} 벡터 저장, dim={embeddings.shape[1]}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
