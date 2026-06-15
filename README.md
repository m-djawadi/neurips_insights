# NeurIPS Insights

Analyze what NeurIPS papers are *about* — themes, conceptual structure, and trends over time — **without downloading a single PDF**. Scrapes only titles and abstracts, embeds them once, and mines that embedding space for a thematic map, cluster relationships, conceptual trends, and semantic search.

The headline stage is **`--analyze`**: it clusters papers in embedding space, picks the papers that *anchor* each cluster (closest to the centroid), builds a graph of which themes sit near each other, and tracks each theme's share over time. The LLM stage then **names** each theme by reading its representative paper titles — not keyword bags — and writes how the themes relate plus a field narrative.

## Why embeddings, not keywords

Keyword counts tell you `training` appears 17 times. They don't tell you that thirty papers are really *the same idea* phrased thirty ways. Embeddings group papers by **meaning**, so paraphrases collapse together, themes emerge that no single keyword names, and "papers like this one" becomes a one-line query.

## Install

```bash
git clone https://github.com/yourname/neurips-insights
cd neurips-insights

pip install -e .              # core: scrape + shallow stats
pip install -e ".[analyze]"   # DEEP analysis: embeddings, clusters, search
pip install -e ".[all]"       # everything incl. legacy BERTopic path
```

For `--llm`, install [Ollama](https://ollama.com): `ollama pull llama3.1`.

## Pipeline

```
--scrape ──► neurips_corpus.jsonl ──► --analyze ──► analysis.json ──► --llm
 (titles+abstracts,                   (embed once, cache to disk:        (Ollama names themes
  streaming, resumable)                clusters, anchors, graph,          from representative
                                       conceptual trends)                 papers; relationships
                                                                          + narrative)
            └──► --search "free text query"   (semantic retrieval)
            └──► --like "a paper title"       (nearest neighbors)
            └──► --stats                       (shallow keyword view, no ML)
```

Embeddings are computed once and cached (`data/emb_*.npy`), keyed on model + corpus size, so every analysis after the first is fast.

## Usage

```bash
# 1. Scrape
neurips-insights --scrape --start 2019 --end 2024

# 2. Deep thematic analysis (auto-picks embedding model & cluster count)
neurips-insights --analyze

# 3. Name themes + relationships + narrative via Ollama
neurips-insights --analyze --llm

# Semantic search — find papers by meaning, not keywords
neurips-insights --search "diffusion models for video generation"

# "Papers like this one"
neurips-insights --like "Denoising Diffusion Probabilistic Models"

# Force a bigger embedding model (GPU recommended)
neurips-insights --analyze --model-name best

# Full pipeline
neurips-insights --scrape --analyze --llm --start 2019 --end 2024
```

### Embedding models (`--model-name`)

| Value | Model | Notes |
|-------|-------|-------|
| `auto` (default) | bge-large if a GPU is visible, else bge-small | sensible default |
| `fast` | `BAAI/bge-small-en-v1.5` (384d) | CPU-friendly, strong |
| `best` | `BAAI/bge-large-en-v1.5` (1024d) | GPU recommended |
| any HF id | e.g. `sentence-transformers/all-mpnet-base-v2` | passed through |

### Flags

| Flag | Purpose |
|------|---------|
| `--scrape` | Fetch titles + abstracts (resumable) |
| `--analyze` | **Deep**: embed → cluster → anchors → graph → trends → `analysis.json` |
| `--llm` | Ollama names themes from representative papers, + relationships + narrative |
| `--search QUERY` | Semantic search over the corpus |
| `--like TITLE` | Papers most similar to one matching `TITLE` |
| `--stats` | Shallow keyword frequencies + trends (no ML) |
| `--topics` | Legacy BERTopic/KMeans path |
| `--model-name` | Embedding model (`auto`/`fast`/`best`/HF id) |
| `--n-topics` | Target cluster count (`0` = auto) |
| `--reps` | Representative papers per cluster |
| `--top-k` | Results for `--search` / `--like` |
| `--force-embed` | Ignore the embedding cache |
| `--model` | Ollama model name |

## What `--analyze` gives you

The output is structured into **five deliverables**, all written to `analysis.json` and printed as a report:

1. **Core themes** — every cluster with size, share, top terms, anchor papers, and a trend arrow.
2. **Evolution of topics** — each theme's early→late share with a growth ratio (e.g. LLM reasoning `3.3% → 23.5%, ×7.0 rising`). Trends use per-year **share**, not raw counts, so conference growth doesn't make everything look like it's rising.
3. **Connected areas** — theme pairs linked by centroid similarity *plus evidence*: their shared vocabulary and the **bridge papers** that sit between them (high similarity to both centroids — often the papers importing a method across areas).
4. **Emerging trends** — an emergence score (recent share × growth acceleration) ranking what's genuinely heating up, not just what's biggest.
5. **Research niches & gaps** — a niche score (small + internally diverse + intermittent = unconsolidated opportunity), shown against a **saturated** list (large + flattening = mature/crowded).

Trends work even with **as few as two years** in the corpus — the trend tag falls back to an early-vs-late share ratio when there aren't enough years for a regression slope, so themes are never left untagged.

### Anti-hallucination LLM synthesis

`--llm` runs in **two passes**. Pass 1 names each cluster *strictly from its own representative paper titles* (strict-JSON reply, low temperature). Pass 2 writes the five-section report over those **names plus the pre-computed numbers and relationships**, with an explicit instruction not to introduce any theme or statistic not in the evidence. This prevents the failure mode where a model labels clusters from their ID numbers and its own priors rather than the actual papers.

## Project structure

```
neurips_insights/
├── cli.py          # argparse entry, wires all stages
├── config.py       # URLs, paths, Ollama settings
├── corpus.py       # streaming JSONL I/O
├── scrape.py       # title+abstract scraper (no PDFs)
├── embeddings.py   # model auto-select + disk-cached vectors
├── analyze.py      # clustering + orchestrates temporal analysis + search
├── temporal.py     # trends, emergence/saturation/niche scoring, relationships
├── llm_deep.py     # two-pass evidence-grounded naming + 5-section report
├── stats.py        # shallow keyword view
├── topics.py       # legacy BERTopic/KMeans
└── llm.py          # legacy keyword-based synthesis
```

## Design notes

- **Embed once, analyze many.** Vectors cache to disk keyed on (model, corpus size). Re-running `--analyze`, `--search`, `--like` reuses them.
- **Anchors over keywords.** Clusters are labeled by the papers nearest their centroid, so the LLM reasons over real titles, not term-frequency noise.
- **bge with the right prompts.** Documents and queries get bge's recommended instruction prefixes for retrieval-quality embeddings.
- **Bounded memory.** Only the `n_docs × dim` float32 matrix is held; the corpus itself streams from disk.

## License

MIT

