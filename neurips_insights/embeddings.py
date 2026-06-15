"""Embedding backend: auto-selects a sensible model, caches vectors to disk.

You embed once; every analysis (clusters, search, neighbors, trends) reuses the
cached matrix. Cache is keyed on (model, corpus length) so it invalidates when
the corpus grows.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Tuple

import numpy as np

from .corpus import iter_corpus, doc_text


# Model tiers. "auto" picks based on whether a CUDA GPU is visible.
# bge models are strong general-purpose retrievers and run fine on CPU at the
# small size; the large variant is worth it only with a GPU.
MODEL_TIERS = {
    "fast": "BAAI/bge-small-en-v1.5",     # 384d, CPU-friendly, great quality/size
    "best": "BAAI/bge-large-en-v1.5",     # 1024d, GPU recommended
}

# bge models expect a short instruction prefix for retrieval-style embedding.
BGE_PREFIX = "Represent this scientific paper for clustering and retrieval: "


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_model(name: str) -> str:
    """Map 'auto'/'fast'/'best'/explicit-id to a concrete model id."""
    if name == "auto":
        return MODEL_TIERS["best"] if _has_gpu() else MODEL_TIERS["fast"]
    if name in MODEL_TIERS:
        return MODEL_TIERS[name]
    return name  # treat as an explicit HF model id


def _cache_path(data_dir: str, model_id: str, n_docs: int) -> str:
    key = hashlib.md5(f"{model_id}:{n_docs}".encode()).hexdigest()[:12]
    safe = model_id.replace("/", "_")
    return os.path.join(data_dir, f"emb_{safe}_{n_docs}_{key}.npy")


def load_corpus_arrays(corpus_path: str):
    """Return (docs, years, titles, urls) as parallel lists."""
    docs, years, titles, urls = [], [], [], []
    for rec in iter_corpus(corpus_path):
        docs.append(doc_text(rec))
        years.append(rec.get("year", 0))
        titles.append(rec.get("title", ""))
        urls.append(rec.get("url", ""))
    return docs, years, titles, urls


def embed_corpus(
    corpus_path: str,
    data_dir: str,
    model_name: str = "auto",
    batch_size: int = 64,
    force: bool = False,
) -> Tuple[np.ndarray, List[str], List[int], List[str], List[str]]:
    """Embed (or load cached) the corpus. Returns (emb, docs, years, titles, urls)."""
    docs, years, titles, urls = load_corpus_arrays(corpus_path)
    model_id = resolve_model(model_name)
    cache = _cache_path(data_dir, model_id, len(docs))

    if os.path.exists(cache) and not force:
        print(f"Loading cached embeddings: {os.path.basename(cache)}")
        emb = np.load(cache)
        if emb.shape[0] == len(docs):
            return emb, docs, years, titles, urls
        print("  cache size mismatch; re-embedding.")

    print(f"Embedding {len(docs)} papers with {model_id} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    inputs = [BGE_PREFIX + d for d in docs] if "bge" in model_id.lower() else docs
    emb = model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine == dot product
    ).astype(np.float32)

    np.save(cache, emb)
    print(f"  cached → {os.path.basename(cache)}")
    return emb, docs, years, titles, urls
