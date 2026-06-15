"""Per-theme technical briefs (Methods / Novelty / Use Cases) for an expert
ML-researcher audience.

Why this is a separate stage from the trend synthesis:

The trend report reasons across ALL themes at once (it needs the global picture
for relationships and rankings). A technical brief is the opposite: it must
reason about ONE theme in depth, grounded strictly in that theme's evidence, so
the model can't borrow details from adjacent clusters. Generating one theme per
call is the structural guarantee against hallucination here — the model only
ever sees the papers it's asked to describe.

Grounding choice: titles alone underdetermine methodology (you can't tell from
"Cold Diffusion" whether it's a sampler, a training scheme, or a theory result).
So we enrich each representative paper with its ABSTRACT pulled from the corpus,
giving the model real technical substance to reason from, and we instruct it to
cite the specific papers it draws each claim from.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .config import OLLAMA_MODEL
from .corpus import iter_corpus


def _build_title_abstract_index(corpus_path: str) -> Dict[str, str]:
    """Map exact title -> abstract, so we can enrich representative papers."""
    idx = {}
    for rec in iter_corpus(corpus_path):
        t = (rec.get("title") or "").strip()
        if t:
            idx[t] = rec.get("abstract", "") or ""
    return idx


def _theme_evidence_block(cluster, abstract_index, max_papers=6, abs_chars=600):
    """Assemble the grounded evidence for one theme: titles + abstracts + terms."""
    lines = []
    terms = ", ".join(cluster.get("top_terms", [])[:10])
    lines.append(f"Distinctive terms: {terms}")
    lines.append(f"Size: {cluster['size']} papers; trend: "
                 f"{cluster['trend']['tag']} (x{cluster['trend']['growth_ratio']}); "
                 f"early->late share {cluster['trend']['early_share']*100:.1f}%"
                 f"->{cluster['trend']['late_share']*100:.1f}%")
    lines.append("Representative papers (title + abstract excerpt):")
    for i, p in enumerate(cluster["representative_papers"][:max_papers], 1):
        title = p["title"]
        abs = abstract_index.get(title, "")
        abs_excerpt = abs[:abs_chars].rsplit(" ", 1)[0] if abs else "(abstract unavailable)"
        lines.append(f"  [{i}] ({p['year']}) {title}\n      {abs_excerpt}")
    return "\n".join(lines)


def _brief_prompt(name, evidence):
    return f"""You are briefing an expert audience of NeurIPS/ICML/ICLR researchers
on a single research theme identified by embedding-based clustering. Below is the
ONLY evidence you may use: the theme's distinctive terms and its representative
papers (titles + abstract excerpts). Reason strictly from this evidence. Do NOT
introduce facts, methods, or applications not supported by these papers. If you
infer something, ground it in a specific paper by its [number].

Theme: {name}

EVIDENCE:
{evidence}

Write three sections, technically precise and concise (this is an expert
audience — assume familiarity with standard ML; skip basics):

**Methods** — The dominant methodologies, architectures, or formulations in this
cluster. Identify the shared technical pattern across the papers, citing [numbers].

**Novelty** — What is new or distinctive here versus prior/adjacent work. Name the
specific methodological shift or conceptual innovation the papers represent, not
generic "this is important". Ground each claim in a paper [number].

**Use Cases** — Realistic applications this line of work enables. Separate
*immediate* (deployable now) from *emerging/future*, and explain WHY these methods
are suited to those uses. Only list applications the evidence actually supports;
if the cluster is purely theoretical, say so and describe downstream relevance
instead of inventing deployments.

No preamble. Start at **Methods**."""


def run_theme_briefs(analysis_path: str, corpus_path: str,
                     model: str = OLLAMA_MODEL, only_ids: List[int] = None,
                     top_n: int = None, out_path: str = None) -> None:
    """Generate a Methods/Novelty/Use-Cases brief per theme.

    only_ids: restrict to specific cluster ids.
    top_n: restrict to the N largest themes (after any only_ids filter).
    out_path: if given, also write all briefs to a markdown file.
    """
    if not os.path.exists(analysis_path):
        raise SystemExit(f"{analysis_path} not found. Run `--analyze` first.")
    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    print("Indexing abstracts for grounding ...")
    abstract_index = _build_title_abstract_index(corpus_path)

    clusters = analysis["clusters"]
    if only_ids:
        clusters = [c for c in clusters if c["id"] in set(only_ids)]
    if top_n:
        clusters = sorted(clusters, key=lambda c: -c["size"])[:top_n]

    # Use names from a prior --llm pass if present, else fall back to terms.
    def theme_name(c):
        return c.get("name") or " ".join(t.capitalize()
                                          for t in c.get("top_terms", [])[:3])

    md_chunks = []
    for c in clusters:
        name = theme_name(c)
        header = f"### {name}  (cluster #{c['id']}, {c['size']} papers)"
        print(f"\n{'='*64}\n{header}\n{'='*64}")
        evidence = _theme_evidence_block(c, abstract_index)
        brief = _ollama_brief(name, evidence, model)
        print(brief)
        md_chunks.append(f"{header}\n\n{brief}\n")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            yrs = analysis.get("years", [])
            span = f"{yrs[0]}–{yrs[1]}" if len(yrs) == 2 else ""
            f.write(f"# NeurIPS Theme Briefs {span}\n\n")
            f.write("\n".join(md_chunks))
        print(f"\nBriefs written to {out_path}")


def _ollama_brief(name, evidence, model):
    # Imported here to reuse the single _ollama implementation + error handling.
    from .llm_deep import _ollama
    return _ollama(_brief_prompt(name, evidence), model, temperature=0.25)
