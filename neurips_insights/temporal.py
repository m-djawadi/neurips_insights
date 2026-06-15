"""Temporal + structural analysis of a clustered paper corpus.

This module turns a set of embedding clusters into the five deliverables:

  1. Core themes          — clusters labeled by their representative papers,
                            with per-year volume.
  2. Evolution of topics  — each theme's trajectory: early vs recent share,
                            growth rate, and a robust trend tag that works
                            even with sparse / non-contiguous years.
  3. Connected areas      — theme-pair relationships backed by *evidence*
                            (shared vocabulary + the bridge papers that sit
                            between two themes), not just centroid cosine.
  4. Emerging trends      — an emergence score ranking themes by recent
                            acceleration, not just size.
  5. Research niches      — underexplored themes: small + internally diverse
                            (sparse coverage) => opportunity.

Design choices (why):
  - We score on YEAR SHARE, not raw counts, because the conference grows
    over time; raw counts would label everything "rising."
  - Trend tag uses early-vs-late share ratio (robust to 2 data points) AND
    a slope when >=3 years exist, so it never collapses to "n/a".
  - Relationships are evidence-based: centroid similarity says "these are
    near", but we attach the actual bridge papers and shared terms so the
    LLM (and you) can verify the link instead of trusting a number.
  - Emergence = recent share weighted by acceleration; saturation = high but
    flattening share. This directly answers "what's growing vs done".
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

import numpy as np


# --------------------------- vocabulary helpers --------------------------- #

_TOKEN = re.compile(r"[a-z][a-z\-]{2,}")
_STOP = set(
    """the of and to for with that this our are from using use used new novel via
    based learning model models method methods approach problem data results show
    propose proposed paper algorithm framework neural network networks deep towards
    can we is on an as by it its their such these those between two more most than
    when each into over under against efficient effective scalable robust general
    improving improved improve toward analysis study learn learning train training
    strong results state achieve performance accuracy empirical experiments benchmark
    framework approach task tasks setting settings problem method methods scalable""".split()
)


def _terms(text: str):
    for w in _TOKEN.findall(text.lower()):
        if w not in _STOP:
            yield w


# ------------------------------- trends ----------------------------------- #

def _year_share(labels: np.ndarray, years: List[int], cluster_ids: List[int]):
    """share[c][year] = fraction of that year's papers in cluster c."""
    yr_total = Counter()
    yr_c = defaultdict(Counter)
    for lab, y in zip(labels, years):
        if lab == -1:
            continue
        yr_total[y] += 1
        yr_c[int(lab)][y] += 1
    all_years = sorted(yr_total)
    share = {}
    for c in cluster_ids:
        share[c] = {y: (yr_c[c][y] / yr_total[y] if yr_total[y] else 0.0)
                    for y in all_years}
    return share, all_years, yr_total


def _trend_tag(year_to_share: Dict[int, float], all_years: List[int]):
    """Robust trend classification that works with as few as 2 years.

    Returns (tag, early_share, late_share, growth_ratio, slope).
    """
    ys = [year_to_share[y] for y in all_years]
    if len(all_years) == 1:
        return "single-year", ys[0], ys[0], 1.0, 0.0

    mid = len(all_years) // 2
    early = np.mean(ys[:mid]) if mid > 0 else ys[0]
    late = np.mean(ys[mid:])
    # growth ratio with smoothing so zero-early doesn't explode
    eps = 1e-4
    ratio = (late + eps) / (early + eps)

    slope = 0.0
    if len(all_years) >= 3:
        xs = np.array(all_years, dtype=float)
        slope = float(np.polyfit(xs - xs.mean(), np.array(ys), 1)[0])

    if ratio >= 1.5:
        tag = "rising"
    elif ratio <= 0.67:
        tag = "declining"
    else:
        tag = "stable"
    return tag, float(early), float(late), float(ratio), slope


# ----------------------------- emergence ---------------------------------- #

def _emergence_score(early, late, ratio, recent_share):
    """Rank how 'hot' a theme is now.

    Combines absolute recent presence (recent_share) with relative growth
    (ratio). A theme that's both growing fast AND already sizeable scores
    highest; a tiny fast-growing theme scores moderately (one to watch).
    Log-dampen the ratio so a 20x jump from near-zero doesn't dominate.
    """
    growth = math.log1p(max(ratio - 1.0, 0.0))      # 0 if not growing
    return float(recent_share * (1.0 + growth))


def _saturation_score(early, late, ratio, share_overall):
    """High overall share but flat/declining growth => saturated/mature."""
    flatness = 1.0 / (1.0 + abs(ratio - 1.0))       # peaks when ratio≈1
    decline = max(0.0, 1.0 - ratio)                 # >0 if shrinking
    return float(share_overall * (flatness + decline))


# --------------------------- niche detection ------------------------------ #

def _niche_score(size, n_total, intra_dispersion, year_coverage, n_years):
    """Small + diffuse + intermittent => underexplored niche / opportunity.

    intra_dispersion: 1 - mean cosine of members to centroid (higher = the
      cluster is loose, i.e. the area isn't consolidated yet).
    year_coverage: fraction of years in which the theme appears at all.
    A real niche is small, not yet methodologically consolidated, and only
    sporadically present — that's where there's room to plant a flag.
    """
    smallness = 1.0 - (size / n_total)
    sparsity = 1.0 - year_coverage
    return float(0.5 * smallness + 0.3 * intra_dispersion + 0.2 * sparsity)


# --------------------------- relationships -------------------------------- #

def _shared_terms(rep_terms_a: Counter, rep_terms_b: Counter, k=6):
    common = []
    for term, _ in (rep_terms_a & rep_terms_b).most_common(k * 2):
        common.append(term)
        if len(common) >= k:
            break
    return common


def _bridge_papers(emb, labels, titles, years, a, b, cent_a, cent_b, top=3):
    """Papers that sit between two clusters: high similarity to BOTH centroids.

    These are the concrete evidence that two themes connect — often the very
    papers that import a method from one area into another.
    """
    idx = np.where((labels == a) | (labels == b))[0]
    if len(idx) == 0:
        return []
    sa = emb[idx] @ cent_a
    sb = emb[idx] @ cent_b
    bridge = np.minimum(sa, sb)            # high only if close to both
    order = idx[np.argsort(-bridge)][:top]
    return [{"title": titles[i], "year": years[i],
             "bridge_strength": round(float(min(emb[i] @ cent_a, emb[i] @ cent_b)), 3)}
            for i in order]


def build_relationships(emb, labels, titles, years, centroids, cluster_terms,
                        cluster_ids, sim_threshold=0.55, top_pairs=20):
    """Evidence-grounded theme links: similarity + shared terms + bridge papers."""
    rels = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            a, b = cluster_ids[i], cluster_ids[j]
            sim = float(centroids[a] @ centroids[b])
            if sim < sim_threshold:
                continue
            shared = _shared_terms(cluster_terms[a], cluster_terms[b])
            bridges = _bridge_papers(emb, labels, titles, years, a, b,
                                     centroids[a], centroids[b])
            rels.append({
                "a": a, "b": b,
                "similarity": round(sim, 3),
                "shared_terms": shared,
                "bridge_papers": bridges,
            })
    rels.sort(key=lambda r: -r["similarity"])
    return rels[:top_pairs]


# ------------------------------ main entry -------------------------------- #

def analyze_temporal(emb, labels, titles, years, docs,
                     cluster_ids, centroids, reps=6) -> Dict[str, Any]:
    """Produce the full structured analysis dict (the five deliverables)."""
    n_total = sum(1 for l in labels if l != -1)
    share, all_years, yr_total = _year_share(labels, years, cluster_ids)
    n_years = len(all_years)
    recent_year = all_years[-1] if all_years else None
    early_cut = all_years[:max(1, n_years // 2)]
    late_cut = all_years[max(1, n_years // 2):] or all_years[-1:]

    # Per-cluster vocabulary from representative docs (for shared-term evidence)
    cluster_terms: Dict[int, Counter] = {}
    cluster_records = []

    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        size = len(idx)
        sims = emb[idx] @ centroids[c]
        order = idx[np.argsort(-sims)]

        # representative papers
        rep_idx = order[:reps]
        rep_papers = [{"title": titles[i], "year": int(years[i]),
                       "similarity": round(float(emb[i] @ centroids[c]), 3)}
                      for i in rep_idx]

        # vocabulary from top-N representative docs
        term_counter = Counter()
        for i in order[: min(40, size)]:
            term_counter.update(set(_terms(docs[i])))
        cluster_terms[c] = term_counter
        top_terms = [t for t, _ in term_counter.most_common(8)]

        # trend
        tag, early, late, ratio, slope = _trend_tag(share[c], all_years)
        recent_share = share[c].get(recent_year, 0.0)
        overall_share = size / n_total

        # scores
        emergence = _emergence_score(early, late, ratio, recent_share)
        saturation = _saturation_score(early, late, ratio, overall_share)
        intra = 1.0 - float(np.mean(sims))            # dispersion
        coverage = sum(1 for y in all_years if share[c][y] > 0) / max(n_years, 1)
        niche = _niche_score(size, n_total, intra, coverage, n_years)

        cluster_records.append({
            "id": c,
            "size": size,
            "share_overall": round(overall_share, 4),
            "top_terms": top_terms,
            "representative_papers": rep_papers,
            "trend": {
                "tag": tag,
                "early_share": round(early, 4),
                "late_share": round(late, 4),
                "growth_ratio": round(ratio, 3),
                "slope": round(slope, 6),
                "by_year": {str(y): round(share[c][y], 4) for y in all_years},
            },
            "scores": {
                "emergence": round(emergence, 4),
                "saturation": round(saturation, 4),
                "niche": round(niche, 4),
                "intra_dispersion": round(intra, 4),
                "year_coverage": round(coverage, 3),
            },
        })

    # relationships
    relationships = build_relationships(
        emb, labels, titles, years, centroids, cluster_terms, cluster_ids)

    # rankings (the actionable part)
    by_emergence = sorted(cluster_records, key=lambda r: -r["scores"]["emergence"])
    by_saturation = sorted(cluster_records, key=lambda r: -r["scores"]["saturation"])
    by_niche = sorted(cluster_records, key=lambda r: -r["scores"]["niche"])

    return {
        "n_papers": n_total,
        "years": [int(all_years[0]), int(all_years[-1])] if all_years else [],
        "papers_per_year": {str(y): yr_total[y] for y in all_years},
        "clusters": sorted(cluster_records, key=lambda r: -r["size"]),
        "relationships": relationships,
        "rankings": {
            "emerging": [{"id": r["id"], "emergence": r["scores"]["emergence"],
                          "terms": r["top_terms"][:5], "tag": r["trend"]["tag"]}
                         for r in by_emergence[:8]],
            "saturated": [{"id": r["id"], "saturation": r["scores"]["saturation"],
                           "terms": r["top_terms"][:5], "tag": r["trend"]["tag"]}
                          for r in by_saturation[:8]],
            "niches": [{"id": r["id"], "niche": r["scores"]["niche"],
                        "size": r["size"], "terms": r["top_terms"][:5]}
                       for r in by_niche[:8]],
        },
    }


# ------------------------------ reporting --------------------------------- #

def print_report(analysis: Dict[str, Any], cluster_label=lambda c: f"#{c['id']}"):
    """Human-readable console version of the five deliverables."""
    yrs = analysis["years"]
    span = f"{yrs[0]}–{yrs[1]}" if len(yrs) == 2 else "?"
    print(f"\n{'='*64}\n NeurIPS thematic analysis — {analysis['n_papers']} papers, {span}\n{'='*64}")

    ppy = analysis["papers_per_year"]
    if len(ppy) >= 2:
        print("\nPapers per year:", ", ".join(f"{y}:{n}" for y, n in ppy.items()))
        print("  (note: trends use per-year SHARE to control for corpus growth)")

    print(f"\n{'—'*64}\n 1. CORE THEMES (by size)\n{'—'*64}")
    for c in analysis["clusters"]:
        t = c["trend"]
        arrow = {"rising": "↑", "declining": "↓", "stable": "→",
                 "single-year": "·"}[t["tag"]]
        print(f"\n{cluster_label(c)}  [{c['size']} papers, {c['share_overall']*100:.1f}%]  "
              f"{arrow} {t['tag']}  (×{t['growth_ratio']:.1f})")
        print(f"   terms: {', '.join(c['top_terms'][:6])}")
        for p in c["representative_papers"][:3]:
            print(f"     - ({p['year']}) {p['title'][:68]}")

    print(f"\n{'—'*64}\n 2. EVOLUTION (early→late share)\n{'—'*64}")
    for c in sorted(analysis["clusters"], key=lambda x: -x["trend"]["growth_ratio"]):
        t = c["trend"]
        print(f"  {cluster_label(c):>28}  {t['early_share']*100:4.1f}% → "
              f"{t['late_share']*100:4.1f}%  ×{t['growth_ratio']:.2f}  {t['tag']}")

    print(f"\n{'—'*64}\n 3. CONNECTED AREAS (evidence-backed)\n{'—'*64}")
    for r in analysis["relationships"][:10]:
        terms = ", ".join(r["shared_terms"][:4]) or "—"
        print(f"  #{r['a']} ↔ #{r['b']}  sim={r['similarity']}  shared: {terms}")
        if r["bridge_papers"]:
            bp = r["bridge_papers"][0]
            print(f"      bridge: ({bp['year']}) {bp['title'][:60]}")

    print(f"\n{'—'*64}\n 4. EMERGING TRENDS (recent acceleration)\n{'—'*64}")
    for e in analysis["rankings"]["emerging"]:
        print(f"  score={e['emergence']:.3f}  #{e['id']}  [{e['tag']}]  "
              f"{', '.join(e['terms'])}")

    print(f"\n{'—'*64}\n 5. RESEARCH NICHES (small + unconsolidated)\n{'—'*64}")
    for nz in analysis["rankings"]["niches"]:
        print(f"  score={nz['niche']:.3f}  #{nz['id']}  [{nz['size']} papers]  "
              f"{', '.join(nz['terms'])}")

    print(f"\n  Saturated (mature, flattening):")
    for s in analysis["rankings"]["saturated"][:5]:
        print(f"    #{s['id']}  {', '.join(s['terms'])}")
