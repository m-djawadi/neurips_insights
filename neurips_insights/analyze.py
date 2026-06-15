"""Embedding-native deep analysis.

Produces a thematic map (clusters + the papers that anchor each), a cluster
relationship graph (which themes sit near each other), conceptual trends over
time, and a semantic-search / nearest-neighbor index. Everything keys off the
cached embedding matrix, so it's fast to re-run.

The output is an `analysis.json` artifact consumed by the LLM stage to write
named themes, relationships, and a narrative.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

from .embeddings import embed_corpus


# ----------------------------- clustering --------------------------------- #

def _auto_k(n_docs: int, requested: int) -> int:
    """Pick a cluster count if the user didn't force one."""
    if requested and requested > 0:
        return min(requested, max(2, n_docs // 2))
    # heuristic: ~sqrt(n/2), clamped to a readable range
    k = int(np.sqrt(n_docs / 2))
    return max(4, min(40, k))


def _cluster(emb: np.ndarray, k: int):
    """Cluster normalized embeddings. Tries HDBSCAN, falls back to KMeans.

    On normalized vectors, KMeans (Euclidean) approximates spherical/cosine
    clustering well and is deterministic + dependency-light.
    """
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, emb.shape[0] // (k * 3) or 5),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(emb)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        if n_found >= 2:
            return labels, "hdbscan"
    except Exception:
        pass

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    return km.fit_predict(emb), "kmeans"


def _centroids(emb: np.ndarray, labels: np.ndarray, cluster_ids: List[int]):
    cents = {}
    for c in cluster_ids:
        mask = labels == c
        v = emb[mask].mean(axis=0)
        n = np.linalg.norm(v)
        cents[c] = v / n if n else v
    return cents


def _representative_papers(emb, labels, titles, urls, years, c, centroid, top=6):
    """Papers closest to a cluster's centroid — the cluster's anchors."""
    idx = np.where(labels == c)[0]
    sims = emb[idx] @ centroid
    order = idx[np.argsort(-sims)][:top]
    return [
        {"title": titles[i], "year": years[i], "url": urls[i],
         "similarity": float(emb[i] @ centroid)}
        for i in order
    ]


# ------------------------- cluster relationships -------------------------- #

def _cluster_graph(centroids: Dict[int, np.ndarray], threshold=0.55):
    """Edges between clusters whose centroids are cosine-similar."""
    ids = sorted(centroids)
    edges = []
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            sim = float(centroids[a] @ centroids[b])
            if sim >= threshold:
                edges.append({"a": a, "b": b, "similarity": round(sim, 3)})
    edges.sort(key=lambda e: -e["similarity"])
    return edges


# ------------------------------ trends ------------------------------------ #

def _trends(labels, years, cluster_ids):
    """Per-cluster share by year, plus a rising/declining slope tag."""
    yr_total = defaultdict(int)
    yr_cluster = defaultdict(lambda: defaultdict(int))
    for lab, y in zip(labels, years):
        if lab == -1:
            continue
        yr_total[y] += 1
        yr_cluster[y][int(lab)] += 1

    all_years = sorted(yr_total)
    trends = {}
    for c in cluster_ids:
        shares = []
        for y in all_years:
            tot = yr_total[y]
            shares.append((y, yr_cluster[y].get(c, 0) / tot if tot else 0.0))
        # simple slope over time as the trend signal
        if len(shares) >= 3:
            xs = np.array([s[0] for s in shares], dtype=float)
            ys = np.array([s[1] for s in shares], dtype=float)
            slope = float(np.polyfit(xs - xs.mean(), ys, 1)[0])
            tag = "rising" if slope > 1e-3 else ("declining" if slope < -1e-3 else "stable")
        else:
            slope, tag = 0.0, "n/a"
        trends[c] = {"by_year": shares, "slope": slope, "tag": tag}
    return trends, all_years


# ------------------------------ main run ---------------------------------- #

def run_analyze(corpus_path: str, data_dir: str, out_path: str,
                model_name: str = "auto", n_topics: int = 0,
                reps: int = 6, force_embed: bool = False) -> Dict[str, Any]:
    emb, docs, years, titles, urls = embed_corpus(
        corpus_path, data_dir, model_name=model_name, force=force_embed)

    k = _auto_k(len(docs), n_topics)
    print(f"\nClustering {len(docs)} papers (target k≈{k}) ...")
    labels, method = _cluster(emb, k)
    cluster_ids = sorted(int(c) for c in set(labels) if c != -1)
    print(f"  method={method}, {len(cluster_ids)} clusters"
          + (f", {(labels==-1).sum()} outliers" if -1 in labels else ""))

    cents = _centroids(emb, labels, cluster_ids)
    trends, all_years = _trends(labels, years, cluster_ids)
    edges = _cluster_graph(cents)

    clusters = []
    for c in cluster_ids:
        size = int((labels == c).sum())
        reps_list = _representative_papers(
            emb, labels, titles, urls, years, c, cents[c], top=reps)
        clusters.append({
            "id": c,
            "size": size,
            "share": round(size / len(docs), 4),
            "trend": trends[c]["tag"],
            "slope": round(trends[c]["slope"], 6),
            "representative_papers": reps_list,
        })

    clusters.sort(key=lambda x: -x["size"])

    # Console summary (structure only; naming happens in the LLM stage)
    print(f"\n{'='*60}\n THEMATIC MAP — {len(clusters)} clusters\n{'='*60}")
    for cl in clusters:
        arrow = {"rising": "↑", "declining": "↓", "stable": "→", "n/a": " "}[cl["trend"]]
        print(f"\n● Cluster {cl['id']}  [{cl['size']} papers, "
              f"{cl['share']*100:.1f}%]  {arrow} {cl['trend']}")
        for p in cl["representative_papers"][:4]:
            print(f"    - ({p['year']}) {p['title'][:72]}")

    if edges:
        print(f"\n{'='*60}\n CLUSTER RELATIONSHIPS (most similar pairs)\n{'='*60}")
        for e in edges[:12]:
            print(f"  {e['a']} ↔ {e['b']}   sim={e['similarity']}")

    payload = {
        "n_papers": len(docs),
        "years": [int(min(all_years)), int(max(all_years))] if all_years else [],
        "method": method,
        "clusters": clusters,
        "relationships": edges,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nAnalysis written to {out_path}")
    return payload


# ------------------------- semantic search -------------------------------- #

def run_search(corpus_path: str, data_dir: str, query: str,
               model_name: str = "auto", top_k: int = 10) -> None:
    """Free-text semantic search over the corpus."""
    emb, docs, years, titles, urls = embed_corpus(
        corpus_path, data_dir, model_name=model_name)

    from .embeddings import resolve_model, BGE_PREFIX
    from sentence_transformers import SentenceTransformer
    model_id = resolve_model(model_name)
    model = SentenceTransformer(model_id)
    q = query if "bge" not in model_id.lower() else \
        "Represent this sentence for searching relevant passages: " + query
    qv = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)

    sims = emb @ qv
    order = np.argsort(-sims)[:top_k]
    print(f"\nTop {top_k} papers for: \"{query}\"\n{'-'*60}")
    for rank, i in enumerate(order, 1):
        print(f"{rank:2d}. ({years[i]}) [{sims[i]:.3f}] {titles[i]}")
        if urls[i]:
            print(f"      {urls[i]}")


def run_neighbors(corpus_path: str, data_dir: str, title_query: str,
                  model_name: str = "auto", top_k: int = 8) -> None:
    """Find papers most similar to the one whose title best matches the query."""
    emb, docs, years, titles, urls = embed_corpus(
        corpus_path, data_dir, model_name=model_name)

    # locate the seed paper by fuzzy title containment
    ql = title_query.lower()
    seed = next((i for i, t in enumerate(titles) if ql in t.lower()), None)
    if seed is None:
        # fall back to embedding the query string itself
        from .embeddings import resolve_model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(resolve_model(model_name))
        qv = model.encode([title_query], normalize_embeddings=True)[0].astype(np.float32)
        sims = emb @ qv
        print(f'No exact title match; showing nearest to query text.')
    else:
        print(f'Seed: ({years[seed]}) {titles[seed]}')
        sims = emb @ emb[seed]
        sims[seed] = -1  # exclude self

    order = np.argsort(-sims)[:top_k]
    print(f"\nMost similar papers\n{'-'*60}")
    for rank, i in enumerate(order, 1):
        print(f"{rank:2d}. ({years[i]}) [{sims[i]:.3f}] {titles[i]}")
