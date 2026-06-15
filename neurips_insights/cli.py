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
    p.add_argument("--stats", action="store_true", help="keyword + trend stats (shallow)")
    p.add_argument("--topics", action="store_true", help="legacy topic modeling")
    p.add_argument("--analyze", action="store_true",
                   help="DEEP: embedding clusters, representative papers, "
                        "cluster graph, conceptual trends")
    p.add_argument("--llm", action="store_true",
                   help="Ollama synthesis (uses --analyze output if present, "
                        "else --topics output)")
    p.add_argument("--search", metavar="QUERY", default=None,
                   help="semantic search over the corpus")
    p.add_argument("--like", metavar="TITLE", default=None,
                   help="find papers similar to one matching this title")
    p.add_argument("--briefs", action="store_true",
                   help="per-theme Methods/Novelty/Use-Cases technical briefs "
                        "(grounded in each theme's representative abstracts)")

    # Shared
    p.add_argument("--data-dir", default="data", help="where files live (default: data/)")
    p.add_argument("--model-name", default="auto",
                   help="embedding model: auto|fast|best or a HF id (default auto)")
    p.add_argument("--force-embed", action="store_true",
                   help="recompute embeddings, ignore cache")

    # Scrape options
    p.add_argument("--start", type=int, default=2020, help="start year (default 2020)")
    p.add_argument("--end", type=int, default=2024, help="end year (default 2024)")
    p.add_argument("--limit-per-year", type=int, default=None,
                   help="cap papers/year (for quick tests)")

    # Topic / analyze options
    p.add_argument("--n-topics", type=int, default=0,
                   help="target cluster count (0=auto)")
    p.add_argument("--no-bertopic", action="store_true",
                   help="legacy --topics: use embeddings+KMeans directly")
    p.add_argument("--reps", type=int, default=6,
                   help="representative papers per cluster (--analyze)")
    p.add_argument("--top-k", type=int, default=10,
                   help="results for --search / --like")

    # Stats options
    p.add_argument("--top-n", type=int, default=40, help="top terms to show")

    # LLM options
    p.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")

    # Briefs options
    p.add_argument("--brief-ids", default=None,
                   help="comma-separated cluster ids to brief (default: all)")
    p.add_argument("--brief-top", type=int, default=None,
                   help="brief only the N largest themes")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    stages = [args.scrape, args.stats, args.topics, args.analyze, args.llm,
              args.search, args.like, args.briefs]
    if not any(stages):
        build_parser().print_help()
        return 1

    os.makedirs(args.data_dir, exist_ok=True)
    corpus_path = os.path.join(args.data_dir, CORPUS_FILE)
    log_path = os.path.join(args.data_dir, SCRAPE_LOG)
    topics_path = os.path.join(args.data_dir, TOPICS_FILE)
    analysis_path = os.path.join(args.data_dir, "analysis.json")

    if args.scrape:
        from .scrape import scrape
        print(f"== SCRAPE {args.start}-{args.end} ==")
        scrape(args.start, args.end, corpus_path, log_path, args.limit_per_year)
        print(f"Corpus now holds {count_records(corpus_path)} papers.")

    if args.stats:
        from .stats import run_stats
        print("\n== STATS (shallow keyword view) ==")
        run_stats(corpus_path, top_n=args.top_n)

    if args.topics:
        from .topics import run_topics
        print("\n== TOPICS (legacy) ==")
        run_topics(corpus_path, args.n_topics or 20, topics_path,
                   use_bertopic=not args.no_bertopic)

    if args.analyze:
        from .analyze import run_analyze
        print("\n== DEEP ANALYSIS ==")
        run_analyze(corpus_path, args.data_dir, analysis_path,
                    model_name=args.model_name, n_topics=args.n_topics,
                    reps=args.reps, force_embed=args.force_embed)

    if args.search:
        from .analyze import run_search
        print("\n== SEMANTIC SEARCH ==")
        run_search(corpus_path, args.data_dir, args.search,
                   model_name=args.model_name, top_k=args.top_k)

    if args.like:
        from .analyze import run_neighbors
        print("\n== PAPERS LIKE THIS ==")
        run_neighbors(corpus_path, args.data_dir, args.like,
                      model_name=args.model_name, top_k=args.top_k)

    if args.briefs:
        from .briefs import run_theme_briefs
        print("\n== THEME BRIEFS (Methods / Novelty / Use Cases) ==")
        ids = None
        if args.brief_ids:
            ids = [int(x) for x in args.brief_ids.split(",") if x.strip()]
        briefs_out = os.path.join(args.data_dir, "theme_briefs.md")
        run_theme_briefs(analysis_path, corpus_path, model=args.model,
                         only_ids=ids, top_n=args.brief_top, out_path=briefs_out)

    if args.llm:
        # Prefer the deep analysis artifact; fall back to legacy topics.json
        print("\n== LLM SYNTHESIS ==")
        if os.path.exists(analysis_path):
            from .llm_deep import run_llm_deep
            run_llm_deep(analysis_path, model=args.model)
        elif os.path.exists(topics_path):
            from .llm import run_llm
            print("(using legacy topics.json; run --analyze for richer synthesis)")
            run_llm(topics_path, model=args.model)
        else:
            print("No analysis found. Run --analyze (preferred) or --topics first.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
