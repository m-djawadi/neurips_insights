"""Command-line interface for NeurIPS Insights.

Stages (each a flag, can combine):
  --scrape   fetch titles + abstracts into a JSONL corpus (resumable)
  --stats    streaming keyword frequencies + year-over-year trends
  --topics   embedding-based topic modeling (BERTopic or KMeans fallback)
  --llm      Ollama synthesis of topics into named themes + the "why"
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .config import (
    CORPUS_FILE, SCRAPE_LOG, TOPICS_FILE, OLLAMA_MODEL,
)
from .corpus import count_records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurips-insights",
        description="Analyze NeurIPS papers from titles + abstracts only (no PDFs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  neurips-insights --scrape --start 2020 --end 2024
  neurips-insights --stats
  neurips-insights --topics --n-topics 20
  neurips-insights --llm --model llama3.1
  neurips-insights --scrape --stats --topics --llm   # full pipeline
""",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Stage flags
    p.add_argument("--scrape", action="store_true", help="scrape titles+abstracts")
    p.add_argument("--stats", action="store_true", help="keyword + trend stats")
    p.add_argument("--topics", action="store_true", help="topic modeling")
    p.add_argument("--llm", action="store_true", help="Ollama theme synthesis")

    # Shared
    p.add_argument("--data-dir", default="data", help="where files live (default: data/)")

    # Scrape options
    p.add_argument("--start", type=int, default=2020, help="start year (default 2020)")
    p.add_argument("--end", type=int, default=2024, help="end year (default 2024)")
    p.add_argument("--limit-per-year", type=int, default=None,
                   help="cap papers/year (for quick tests)")

    # Topic options
    p.add_argument("--n-topics", type=int, default=20,
                   help="target topic count (default 20; 0=auto for BERTopic)")
    p.add_argument("--no-bertopic", action="store_true",
                   help="skip BERTopic, use embeddings+KMeans directly")

    # Stats options
    p.add_argument("--top-n", type=int, default=40, help="top terms to show")

    # LLM options
    p.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not any([args.scrape, args.stats, args.topics, args.llm]):
        build_parser().print_help()
        return 1

    os.makedirs(args.data_dir, exist_ok=True)
    corpus_path = os.path.join(args.data_dir, CORPUS_FILE)
    log_path = os.path.join(args.data_dir, SCRAPE_LOG)
    topics_path = os.path.join(args.data_dir, TOPICS_FILE)

    if args.scrape:
        from .scrape import scrape
        print(f"== SCRAPE {args.start}-{args.end} ==")
        scrape(args.start, args.end, corpus_path, log_path, args.limit_per_year)
        print(f"Corpus now holds {count_records(corpus_path)} papers.")

    if args.stats:
        from .stats import run_stats
        print("\n== STATS ==")
        run_stats(corpus_path, top_n=args.top_n)

    if args.topics:
        from .topics import run_topics
        print("\n== TOPICS ==")
        run_topics(corpus_path, args.n_topics, topics_path,
                   use_bertopic=not args.no_bertopic)

    if args.llm:
        from .llm import run_llm
        print("\n== LLM SYNTHESIS ==")
        run_llm(topics_path, model=args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
