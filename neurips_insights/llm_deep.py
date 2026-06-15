"""Evidence-grounded LLM synthesis over the temporal analysis artifact.

Why this is structured the way it is:

The earlier version let the model free-associate from cluster *numbers*, which
produced confident nonsense (e.g. labeling unrelated clusters "Federated
Learning"). The fix is to (a) feed only verifiable evidence -- representative
paper TITLES, extracted TERMS, and PRE-COMPUTED scores -- and (b) make naming a
two-pass process: first the model assigns a name to each cluster from its own
papers, then we substitute those names into the relationship/trend sections so
it never reasons about a cluster it hasn't seen the papers for.

The output maps 1:1 to the five requested deliverables.
"""

from __future__ import annotations

import json
import os

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL


def _ollama(prompt: str, model: str, temperature: float = 0.2) -> str:
    """Low temperature: we want grounded labeling, not creative writing."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=900,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Try `ollama serve` and `ollama pull {model}`."
        )


def _name_clusters(clusters, model):
    """Pass 1: name each cluster strictly from its representative papers.

    Returns {id: name}. We parse a strict JSON reply so names can be reused.
    """
    blocks = []
    for c in clusters:
        papers = "\n".join(f"      - {p['title']}"
                           for p in c["representative_papers"][:5])
        terms = ", ".join(c["top_terms"][:8])
        blocks.append(
            f"  Cluster {c['id']}:\n    terms: {terms}\n    papers:\n{papers}"
        )

    prompt = f"""You label research clusters from a top ML conference. For EACH
cluster below, read its representative paper TITLES and TERMS and assign a
precise 2-5 word research-area name. Base the name ONLY on the evidence shown --
do not invent areas that aren't supported by these titles.

Return STRICT JSON only, no prose: an object mapping the cluster id (as a string)
to its name. Example: {{"0": "Offline Reinforcement Learning", "1": "Diffusion Models"}}

CLUSTERS:
{chr(10).join(blocks)}
"""
    raw = _ollama(prompt, model, temperature=0.1)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        names = json.loads(raw[start:end])
        return {int(k): v for k, v in names.items()}
    except Exception:
        return {c["id"]: ", ".join(c["top_terms"][:3]).title() for c in clusters}


def _synthesize(analysis, names, model):
    """Pass 2: narrative over named themes + computed trends/relationships."""
    def nm(cid):
        return names.get(cid, f"#{cid}")

    yrs = analysis.get("years", [])
    span = f"{yrs[0]}-{yrs[1]}" if len(yrs) == 2 else "the period"

    evo = []
    for c in sorted(analysis["clusters"],
                    key=lambda x: -x["trend"]["growth_ratio"]):
        t = c["trend"]
        evo.append(f"  {nm(c['id'])}: {t['early_share']*100:.1f}% -> "
                   f"{t['late_share']*100:.1f}% (x{t['growth_ratio']:.2f}, {t['tag']})")

    rels = []
    for r in analysis["relationships"][:10]:
        bridge = r["bridge_papers"][0]["title"] if r["bridge_papers"] else "-"
        shared = ", ".join(r["shared_terms"][:4]) or "-"
        rels.append(f"  {nm(r['a'])} <-> {nm(r['b'])} (sim {r['similarity']}); "
                    f"shared: {shared}; bridge paper: {bridge}")

    emerging = "\n".join(
        f"  {nm(e['id'])} (emergence {e['emergence']:.3f}, {e['tag']})"
        for e in analysis["rankings"]["emerging"])
    niches = "\n".join(
        f"  {nm(n['id'])} ({n['size']} papers, niche {n['niche']:.3f})"
        for n in analysis["rankings"]["niches"])
    saturated = "\n".join(
        f"  {nm(s['id'])} (saturation {s['saturation']:.3f}, {s['tag']})"
        for s in analysis["rankings"]["saturated"][:6])

    prompt = f"""You are a senior ML researcher writing an evidence-based trend
report on a conference over {span} ({analysis['n_papers']} papers). ALL theme
names, numbers, and relationships below were computed from the data -- treat them
as ground truth and DO NOT introduce themes or statistics not listed here.

Write five concise sections with these exact headers:

# 1. CORE THEMES
The dominant research areas and what each is about (one line each, top ~8 by size).

# 2. EVOLUTION OF TOPICS
Using the early->late shares, say which areas grew, shrank, or held steady, and
by roughly how much. Distinguish early-period vs recent character of the field.

# 3. CONNECTED / LAYERED AREAS
Using the relationship pairs (shared terms + bridge papers), explain which areas
build on or feed into each other, and what the bridge papers suggest about how
methods transfer. Reference the named themes.

# 4. EMERGING TRENDS
Interpret the emergence ranking: what's accelerating now and why it likely matters.

# 5. RESEARCH NICHES & GAPS
Interpret the niche ranking and the saturated list together: where is the field
crowded (saturated) vs where is there underexplored room (niches)? Give 3-4
concrete, actionable directions a researcher could pursue, each tied to a named
theme or a gap between two themes.

Be specific and grounded. No preamble.

=== CORE THEMES (named, by size) ===
{chr(10).join('  ' + nm(c['id']) + f" [{c['size']} papers]" for c in analysis['clusters'][:12])}

=== EVOLUTION (early->late share) ===
{chr(10).join(evo)}

=== RELATIONSHIPS (evidence-backed) ===
{chr(10).join(rels)}

=== EMERGING (ranked) ===
{emerging}

=== SATURATED (mature/flattening) ===
{saturated}

=== NICHES (small + unconsolidated) ===
{niches}
"""
    return _ollama(prompt, model, temperature=0.3)


def run_llm_deep(analysis_path: str, model: str = OLLAMA_MODEL) -> None:
    if not os.path.exists(analysis_path):
        raise SystemExit(f"{analysis_path} not found. Run `--analyze` first.")
    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    print(f"Pass 1/2: naming themes from representative papers ({model}) ...")
    names = _name_clusters(analysis["clusters"], model)

    for c in analysis["clusters"]:
        c["name"] = names.get(c["id"], "")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print("Named themes:")
    for c in analysis["clusters"][:12]:
        print(f"  #{c['id']:>2} -> {c['name']}")

    print(f"\nPass 2/2: synthesizing trend report ...\n")
    print(_synthesize(analysis, names, model))
