"""Embedding-based topic modeling (no TF-IDF as the primary signal).

Primary path: BERTopic (sentence-transformer embeddings + UMAP + HDBSCAN).
Fallback path: sentence-transformer embeddings + MiniBatchKMeans, with
c-TF-IDF-style keyword extraction per cluster for labels.

Embeddings are computed in batches so peak memory stays bounded. The doc
texts themselves are streamed from disk; only the embedding matrix
(n_docs x dim, float32) is held — ~50k docs x 384 dims ≈ 75 MB.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .corpus import iter_corpus, doc_text


EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, 384-dim


def _load_docs(corpus_path: str):
    docs, years, titles = [], [], []
    for rec in iter_corpus(corpus_path):
        docs.append(doc_text(rec))
        years.append(rec.get("year", 0))
        titles.append(rec.get("title", ""))
    return docs, years, titles


def _embed(docs, batch_size=256):
    """Batch-encode docs to a float32 matrix. Bounded peak memory."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def _topic_trends(years, labels, n_topics, topics_meta, out_path):
    """Compute dominant-topic share per year and persist topic metadata."""
    import numpy as np

    yr_topic = defaultdict(lambda: defaultdict(int))
    yr_total = defaultdict(int)
    for y, lab in zip(years, labels):
        yr_total[y] += 1
        yr_topic[y][int(lab)] += 1

    print("\nTopic prevalence by year (dominant topic per year)")
    for y in sorted(yr_topic):
        counts = yr_topic[y]
        top = max(counts, key=counts.get)
        share = 100.0 * counts[top] / max(yr_total[y], 1)
        label = topics_meta.get(top, {}).get("label", f"#{top}")
        print(f"  {y}: {label}  ({share:.0f}%)")

    # Persist for the --llm stage
    payload = {
        "topics": topics_meta,
        "year_distribution": {
            str(y): {str(k): v for k, v in yr_topic[y].items()} for y in yr_topic
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nTopic metadata written to {out_path}")


def _keywords_per_cluster(docs, labels, n_topics, top_k=10):
    """c-TF-IDF-style: most distinctive terms per cluster for labeling."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=5, max_features=20000)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    # Aggregate term counts per cluster
    meta = {}
    import scipy.sparse as sp

    for k in range(n_topics):
        mask = np.array(labels) == k
        if mask.sum() == 0:
            meta[k] = {"label": f"topic_{k}", "keywords": [], "size": 0}
            continue
        cluster_counts = np.asarray(X[mask].sum(axis=0)).ravel()
        total_counts = np.asarray(X.sum(axis=0)).ravel()
        # c-TF-IDF: term freq in cluster scaled by inverse overall frequency
        ctfidf = cluster_counts / (1.0 + total_counts)
        top_idx = ctfidf.argsort()[-top_k:][::-1]
        kws = [w for w in vocab[top_idx]]
        meta[k] = {
            "label": ", ".join(kws[:4]),
            "keywords": kws,
            "size": int(mask.sum()),
        }
    return meta


def run_topics(corpus_path: str, n_topics: int, out_path: str,
               use_bertopic: bool = True) -> None:
    print("Loading corpus...")
    docs, years, titles = _load_docs(corpus_path)
    print(f"  {len(docs)} documents")

    if use_bertopic:
        try:
            _run_bertopic(docs, years, n_topics, out_path)
            return
        except ImportError:
            print("BERTopic not installed; falling back to embeddings + KMeans.")
            print("(`pip install bertopic` for the richer path.)")

    _run_kmeans(docs, years, n_topics, out_path)


def _run_bertopic(docs, years, n_topics, out_path):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(EMBED_MODEL)
    # nr_topics reduces to a target count; None lets HDBSCAN decide.
    topic_model = BERTopic(
        embedding_model=embed_model,
        nr_topics=n_topics if n_topics > 0 else "auto",
        calculate_probabilities=False,
        verbose=True,
    )
    labels, _ = topic_model.fit_transform(docs)

    info = topic_model.get_topic_info()
    print("\nDiscovered topics")
    meta = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # outliers
        words = [w for w, _ in topic_model.get_topic(tid)][:10]
        meta[tid] = {"label": ", ".join(words[:4]), "keywords": words,
                     "size": int(row["Count"])}
        print(f"  #{tid:3d} [{row['Count']:5d}]  {', '.join(words[:8])}")

    n_eff = max(meta) + 1 if meta else 0
    _topic_trends(years, labels, n_eff, meta, out_path)


def _run_kmeans(docs, years, n_topics, out_path):
    from sklearn.cluster import MiniBatchKMeans

    print("Embedding documents (batched)...")
    emb = _embed(docs)
    print("Clustering (MiniBatchKMeans)...")
    km = MiniBatchKMeans(n_clusters=n_topics, random_state=0, batch_size=1024,
                         n_init=3)
    labels = km.fit_predict(emb)

    meta = _keywords_per_cluster(docs, labels, n_topics)
    print("\nDiscovered topics (embeddings + KMeans)")
    for k in sorted(meta, key=lambda k: -meta[k]["size"]):
        m = meta[k]
        print(f"  #{k:3d} [{m['size']:5d}]  {', '.join(m['keywords'][:8])}")

    _topic_trends(years, labels, n_topics, meta, out_path)
