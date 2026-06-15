"""Ollama-powered synthesis: turn topic keyword-lists into named themes + the
"why". Feeds only compact topic metadata to the LLM (never raw abstracts in
bulk), so this stays cheap regardless of corpus size.
"""

from __future__ import annotations

import json
import os

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL


def _ollama(prompt: str, model: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "Could not reach Ollama at "
            f"{OLLAMA_URL}.\nIs it running? Try `ollama serve` and "
            f"`ollama pull {model}`."
        )


def _trend_summary(year_dist, topics):
    """Build a compact text summary of how each topic's share moves over time."""
    years = sorted(int(y) for y in year_dist)
    if not years:
        return ""
    lines = []
    for tid, meta in topics.items():
        tid_i = int(tid)
        per_year = []
        for y in years:
            counts = year_dist[str(y)]
            total = sum(counts.values())
            share = counts.get(str(tid_i), 0) / total if total else 0
            per_year.append(share)
        if len(per_year) >= 4:
            early = sum(per_year[: len(per_year) // 2])
            late = sum(per_year[len(per_year) // 2 :])
            arrow = "rising" if late > early * 1.2 else (
                "falling" if late < early * 0.8 else "stable")
        else:
            arrow = "n/a"
        lines.append(f"- {meta['label']} ({arrow})")
    return "\n".join(lines)


def run_llm(topics_path: str, model: str = OLLAMA_MODEL) -> None:
    if not os.path.exists(topics_path):
        raise SystemExit(
            f"{topics_path} not found. Run `--topics` first to generate it."
        )
    with open(topics_path, encoding="utf-8") as f:
        data = json.load(f)

    topics = data["topics"]
    year_dist = data.get("year_distribution", {})

    # Compose a single compact prompt from keyword lists only.
    topic_lines = []
    for tid, meta in sorted(topics.items(), key=lambda kv: -kv[1]["size"]):
        kws = ", ".join(meta["keywords"][:10])
        topic_lines.append(f"Topic {tid} (n={meta['size']}): {kws}")

    trends = _trend_summary(year_dist, topics)

    prompt = f"""You are analyzing research themes in NeurIPS (a top machine
learning conference), extracted as keyword clusters from paper abstracts.

For EACH topic below, output exactly:
  - Name: a 3-6 word human-readable theme name
  - What: one sentence on the research problem it addresses
  - Why: one sentence on why this area matters or grew

Then add a final section "BIG PICTURE" (3-4 sentences) describing the overall
trajectory of the field given which themes are rising vs falling.

Be concise. No preamble.

TOPICS:
{chr(10).join(topic_lines)}

TREND DIRECTION (rising/falling/stable):
{trends}
"""

    print(f"Querying Ollama ({model})...\n")
    print(_ollama(prompt, model))
