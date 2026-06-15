# NeurIPS Insights

Analyze what NeurIPS papers are *about* — main themes, frequent topics, and trends over time — **without downloading a single PDF**. The pipeline scrapes only titles and abstracts, stores them as a compact streaming corpus, and mines them with three escalating layers: cheap keyword stats, embedding-based topic modeling, and optional LLM synthesis via Ollama.

Built for low storage and low memory: the corpus is tens of MB of text (vs. hundreds of GB of PDFs), records stream one at a time, and embeddings are computed in bounded batches.

## Why titles + abstracts only

A title plus abstract is ~1–2 KB. The full proceedings (2006–present, ~50k papers) is well under 100 MB of text — small enough to re-analyze as often as you like, and small enough that scraping is polite and fast. PDFs add hundreds of GB and almost nothing for thematic analysis.

## Install

```bash
git clone https://github.com/yourname/neurips-insights
cd neurips-insights

# Core only (scrape + stats) — pure Python, tiny footprint
pip install -e .

# Add embedding-based topic modeling (KMeans fallback path)
pip install -e ".[topics]"

# Add the full BERTopic path (UMAP + HDBSCAN)
pip install -e ".[bertopic]"

# Everything
pip install -e ".[all]"
```

For `--llm`, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.1
```

## Pipeline at a glance

```
--scrape ──► data/neurips_corpus.jsonl ──┬──► --stats   (keywords + trends, no ML)
   (titles + abstracts only,             ├──► --topics  (embeddings ► clusters ► data/topics.json)
    streaming, resumable)                └──► --llm     (Ollama reads topics.json ► named themes + "why")
```

Scraping and analysis are decoupled. Scrape once; analyze many times offline.

## Usage

```bash
# 1. Scrape a range of years (resumable — safe to Ctrl-C and rerun)
neurips-insights --scrape --start 2020 --end 2024

# 2. Cheapest insight: top terms + rising/falling trends (no ML deps)
neurips-insights --stats

# 3. Topic modeling (BERTopic if installed, else embeddings + KMeans)
neurips-insights --topics --n-topics 20

# 4. Turn topic keyword-clusters into named themes + the "why" (Ollama)
neurips-insights --llm --model llama3.1

# Run the whole pipeline in one go
neurips-insights --scrape --stats --topics --llm --start 2018 --end 2024

# Quick smoke test: only 20 papers/year
neurips-insights --scrape --start 2023 --end 2023 --limit-per-year 20 --stats
```

### Flags

| Flag | Purpose |
|------|---------|
| `--scrape` | Fetch titles + abstracts into the JSONL corpus |
| `--stats` | Streaming keyword frequencies + year-over-year trend deltas |
| `--topics` | Embedding-based topic modeling, writes `topics.json` |
| `--llm` | Ollama synthesis of `topics.json` into named themes |
| `--start / --end` | Year range to scrape (default 2020–2024) |
| `--limit-per-year` | Cap papers/year for fast testing |
| `--n-topics` | Target topic count (`0` = auto, BERTopic only) |
| `--no-bertopic` | Force the embeddings + KMeans fallback |
| `--top-n` | How many top terms `--stats` prints |
| `--model` | Ollama model name |
| `--data-dir` | Where corpus / logs / topics live (default `data/`) |

## What you get

**`--stats`** prints papers-per-year, the top terms overall, and the fastest **rising** and **falling** terms (split at the midpoint year) — enough to see classical ML (kernels, boosting, graphical models) give way to transformers, diffusion, and LLMs.

**`--topics`** prints each discovered cluster with its top keywords and size, then the dominant topic per year, and writes `topics.json` (topic keywords + per-year distribution) for the LLM stage.

**`--llm`** feeds only the compact keyword clusters (never raw abstracts in bulk) to Ollama and prints, per topic, a **Name / What / Why**, plus a **BIG PICTURE** paragraph on the field's trajectory.

## Project structure

```
neurips-insights/
├── neurips_insights/
│   ├── __init__.py
│   ├── cli.py        # argparse entry point, wires the four stages
│   ├── config.py     # URLs, paths, delays, Ollama settings
│   ├── corpus.py     # streaming JSONL read/write (constant memory)
│   ├── scrape.py     # title+abstract scraper (resumable, no PDFs)
│   ├── stats.py      # keyword frequencies + trend deltas (pure Python)
│   ├── topics.py     # embeddings ► BERTopic / KMeans ► topics.json
│   └── llm.py        # Ollama synthesis of topics into themes
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Design notes

- **Streaming everywhere.** The scraper flushes each record to disk immediately (crash-safe, resumable via a hash log). Stats stream the corpus line-by-line. Only topic modeling holds a matrix in RAM — and that's just `n_docs × 384` float32s (~75 MB for 50k papers).
- **Embeddings over TF-IDF for topics.** Clusters form in semantic space, so paraphrases group together; TF-IDF is used only *after* clustering, c-TF-IDF style, to label each cluster with distinctive terms.
- **The LLM never sees the raw corpus.** It reads compact keyword lists from `topics.json`, so the synthesis cost is constant regardless of how many papers you scraped.
- **Polite scraping.** Randomized 1.5–3.5s delay between requests; metadata pulled from `citation_*` meta tags with a longest-paragraph fallback for older pages.

## License

MIT
