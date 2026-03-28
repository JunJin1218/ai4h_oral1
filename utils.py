from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class SimilarResult:
    vector_id: int
    score: float
    vector: np.ndarray
    file_name: str | None = None


def _normalize_metric_name(similarity_type: str) -> str:
    metric = similarity_type.lower()
    if metric in {"l2", "euclidean"}:
        return "l2"
    if metric in {"ip", "inner_product", "dot"}:
        return "ip"
    if metric in {"cosine", "cos", "cos_sim"}:
        return "cosine"
    raise ValueError(
        "similarity_type must be one of: l2, euclidean, ip, inner_product, dot, "
        "cosine, cos, cos_sim"
    )


def get_vector_id_by_file_name(
    file_name: str,
    db_path: str | Path = "data/sqlite/ai4h.db",
) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT vector_id FROM image_db WHERE file_name = ?", (file_name,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"file_name not found in DB: {file_name}")
    return int(row[0])


def get_file_name_by_vector_id(
    vector_id: int,
    db_path: str | Path = "data/sqlite/ai4h.db",
) -> str | None:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM image_db WHERE vector_id = ?", (int(vector_id),))
        row = cur.fetchone()
    return None if row is None else str(row[0])


def reconstruct_vector_by_id(
    vector_id: int,
    index_path: str | Path = "data/faiss/embeddings.index",
) -> np.ndarray:
    index = faiss.read_index(str(index_path))
    return index.reconstruct(int(vector_id)).astype(np.float32, copy=False)


def _load_all_vectors(index: faiss.Index) -> tuple[np.ndarray, np.ndarray]:
    ntotal = int(index.ntotal)
    if hasattr(index, "id_map"):
        ids = faiss.vector_to_array(index.id_map).astype(np.int64, copy=False)
    else:
        ids = np.arange(ntotal, dtype=np.int64)
    vectors = np.vstack([index.reconstruct(int(vec_id)) for vec_id in ids]).astype(
        np.float32, copy=False
    )
    return ids, vectors


def search_similar_with_metadata(
    query_vector: Any,
    similarity_type: str = "l2",
    top_n: int = 5,
    index_path: str | Path = "data/faiss/embeddings.index",
    db_path: str | Path = "data/sqlite/ai4h.db",
    exclude_vector_id: int | None = None,
) -> list[SimilarResult]:
    """
    Search similar vectors and return vector_id/score/vector/file_name together.
    """
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    metric = _normalize_metric_name(similarity_type)
    index = faiss.read_index(str(index_path))
    ntotal = int(index.ntotal)
    if ntotal == 0:
        return []

    dim = int(index.d)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.shape[0] != dim:
        raise ValueError(f"query dim mismatch: got {query.shape[0]}, expected {dim}")

    # Keep enough candidates after self-match exclusion.
    fetch_k = min(ntotal, top_n + (1 if exclude_vector_id is not None else 0))
    ids_all, vectors_all = _load_all_vectors(index)

    if metric == "l2":
        search_index = faiss.IndexFlatL2(dim)
        corpus = vectors_all
        q = query.reshape(1, -1)
    elif metric == "ip":
        search_index = faiss.IndexFlatIP(dim)
        corpus = vectors_all
        q = query.reshape(1, -1)
    else:
        search_index = faiss.IndexFlatIP(dim)
        corpus = vectors_all.copy()
        q = query.reshape(1, -1).copy()
        faiss.normalize_L2(corpus)
        faiss.normalize_L2(q)

    search_index.add(corpus)
    scores, local_indices = search_index.search(q, fetch_k)

    file_name_map: dict[int, str] = {}
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT vector_id, file_name FROM image_db")
        for vec_id, file_name in cur.fetchall():
            file_name_map[int(vec_id)] = str(file_name)

    results: list[SimilarResult] = []
    for rank, local_i in enumerate(local_indices[0]):
        if local_i == -1:
            continue
        vec_id = int(ids_all[local_i])
        if exclude_vector_id is not None and vec_id == exclude_vector_id:
            continue
        results.append(
            SimilarResult(
                vector_id=vec_id,
                score=float(scores[0][rank]),
                vector=vectors_all[local_i].copy(),
                file_name=file_name_map.get(vec_id),
            )
        )
        if len(results) >= top_n:
            break
    return results


def find_similar_vectors(
    query_vector: Any,
    similarity_type: str = "l2",
    top_n: int = 5,
    index_path: str | Path = "data/faiss/embeddings.index",
) -> list[np.ndarray]:
    """
    Return top-N similar vectors from a FAISS index.
    """
    results = search_similar_with_metadata(
        query_vector=query_vector,
        similarity_type=similarity_type,
        top_n=top_n,
        index_path=index_path,
    )
    return [r.vector for r in results]


def find_similar_vectors_by_file_name(
    file_name: str,
    similarity_type: str = "l2",
    top_n: int = 10,
    db_path: str | Path = "data/sqlite/ai4h.db",
    index_path: str | Path = "data/faiss/embeddings.index",
) -> list[np.ndarray]:
    """
    Resolve query vector by file_name (via SQLite), then call find_similar_vectors().
    """
    vector_id = get_vector_id_by_file_name(file_name=file_name, db_path=db_path)
    query_vector = reconstruct_vector_by_id(vector_id=vector_id, index_path=index_path)
    return find_similar_vectors(
        query_vector=query_vector,
        similarity_type=similarity_type,
        top_n=top_n,
        index_path=index_path,
    )


def find_similar_by_file_name_with_metadata(
    file_name: str,
    similarity_type: str = "l2",
    top_n: int = 10,
    db_path: str | Path = "data/sqlite/ai4h.db",
    index_path: str | Path = "data/faiss/embeddings.index",
    exclude_self: bool = True,
) -> list[SimilarResult]:
    """
    file_name -> query vector -> similar candidates with (id, score, vector, file_name).
    """
    query_id = get_vector_id_by_file_name(file_name=file_name, db_path=db_path)
    query_vector = reconstruct_vector_by_id(vector_id=query_id, index_path=index_path)
    return search_similar_with_metadata(
        query_vector=query_vector,
        similarity_type=similarity_type,
        top_n=top_n,
        index_path=index_path,
        db_path=db_path,
        exclude_vector_id=query_id if exclude_self else None,
    )
