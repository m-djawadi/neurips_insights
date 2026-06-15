"""Streaming JSONL corpus I/O. Never loads the whole corpus into memory."""

from __future__ import annotations

import json
import os
from typing import Iterator, Dict, Any


def iter_corpus(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one paper record at a time. Constant memory."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found: {path}\nRun `neurips-insights --scrape` first."
        )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_record(path: str, record: Dict[str, Any]) -> None:
    """Append a single record and flush immediately (crash-safe)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def count_records(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def doc_text(record: Dict[str, Any]) -> str:
    """Combine title + abstract into a single analysis document."""
    title = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    return f"{title}. {abstract}".strip()
