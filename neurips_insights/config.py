"""Shared configuration and constants."""

from __future__ import annotations

BASE = "https://papers.nips.cc"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) neurips-insights/0.1"}

# Default file paths (relative to --data-dir)
CORPUS_FILE = "neurips_corpus.jsonl"
SCRAPE_LOG = "scraped_hashes.log"
TOPICS_FILE = "topics.json"

# Polite scraping delay (seconds), uniform random in this range
DELAY_MIN = 1.5
DELAY_MAX = 3.5

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
