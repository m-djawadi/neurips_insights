"""LLM synthesis over the deep-analysis artifact.

Unlike the keyword-based stage, this feeds the LLM the *representative paper
titles* of each cluster, the cluster's size/trend, and the relationship graph.
The model names each theme, explains what binds the papers, describes how themes
relate, and writes a field-level narrative.
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
            timeout=600,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Try `ollama serve` and `ollama pull {model}`."
        )


def _cluster_block(clusters):
    lines = []
    for cl in clusters:
        papers = "; ".join(
            f"({p['year']}) {p['title']}" for p in cl["representative_papers"][:5]
        )
        lines.append(
            f"Cluster {cl['id']} — {cl['size']} papers ({cl['share']*100:.1f}%), "
            f"trend={cl['trend']}\n  Representative: {papers}"
        )
    return "\n".join(lines)


def _rel_block(relationships, clusters):
    if not relationships:
        return "(no strong cross-cluster similarities)"
    by_id = {c["id"]: c for c in clusters}
    lines = []
    for e in relationships[:15]:
        a, b = e["a"], e["b"]
        if a in by_id and b in by_id:
            lines.append(f"Cluster {a} ↔ Cluster {b} (similarity {e['similarity']})")
    return "\n".join(lines)


def run_llm_deep(analysis_path: str, model: str = OLLAMA_MODEL) -> None:
    if not os.path.exists(analysis_path):
        raise SystemExit(
            f"{analysis_path} not found. Run `--analyze` first."
        )
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)

    clusters = data["clusters"]
    rels = data.get("relationships", [])
    yrs = data.get("years", [])
    span = f"{yrs[0]}–{yrs[1]}" if len(yrs) == 2 else "the corpus"

    prompt = f"""You are a senior ML researcher analyzing themes at NeurIPS over
{span} ({data['n_papers']} papers). Each cluster below is defined by its most
representative papers (closest to the cluster centroid in embedding space) — read
the TITLES to infer the theme. Do NOT just list keywords.

For EACH cluster output:
  ## <3-6 word theme name>  (Cluster <id>)
  - **What binds these papers:** one or two sentences on the shared research problem/approach.
  - **Trajectory:** interpret the trend tag in plain language.

Then two sections:

  # HOW THE THEMES RELATE
  Use the relationship pairs to explain which themes are methodologically or
  topically adjacent, and why (2-4 sentences). Reference theme names, not IDs.

  # FIELD NARRATIVE
  4-6 sentences: the big picture. What is NeurIPS centered on in this period,
  which directions are ascending vs fading, and what that implies about where
  the field is heading.

Be specific and concise. No preamble.

CLUSTERS:
{_cluster_block(clusters)}

RELATIONSHIPS (cosine similarity between cluster centroids):
{_rel_block(rels, clusters)}
"""

    print(f"Querying Ollama ({model}) for theme synthesis ...\n")
    print(_ollama(prompt, model))
