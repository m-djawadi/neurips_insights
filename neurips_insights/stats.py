"""Cheap insights: keyword frequencies and year-over-year trends.

Pure Python, streaming, no ML dependencies. Answers "what is NeurIPS about
and how has it shifted" before any heavy machinery is needed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .corpus import iter_corpus, doc_text

# Domain-aware stopwords: generic English + ML-paper filler that drowns signal.
STOP = set(
    """the of and to a in for we is on with that this our as are by an be can will
    which using use used from at it its model models method methods approach approaches
    problem problems learning data results result show shows propose proposed paper
    algorithm algorithms function functions based new novel via these those their such
    both also more most than then when each other into over under between two one three
    however thus therefore given several many much may might can could would should
    while where what whwhy how here there been being have has had not no yes do does
    set sets large small high low first second third number numbers case cases time times
    work works study studies present presents provide provides obtain obtained achieve
    achieves achieved demonstrate demonstrates significant significantly state art
    different various general specific particular order means well good better best
    improvement improvements benchmark benchmarks dataset datasets strong empirical
    across settings setting baseline baselines task tasks framework propose novel""".split()
)

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


def tokens(text: str):
    for w in TOKEN_RE.findall(text.lower()):
        if w not in STOP and not w.startswith("-") and not w.endswith("-"):
            yield w


def run_stats(corpus_path: str, top_n: int = 40) -> None:
    overall = Counter()
    by_year = defaultdict(Counter)
    year_counts = Counter()

    for rec in iter_corpus(corpus_path):  # streaming, constant memory
        year = rec.get("year", 0)
        year_counts[year] += 1
        for t in tokens(doc_text(rec)):
            overall[t] += 1
            by_year[year][t] += 1

    total = sum(year_counts.values())
    print(f"\n{'='*56}\n NeurIPS corpus: {total} papers, "
          f"{min(year_counts)}–{max(year_counts)}\n{'='*56}")

    print("\nPapers per year")
    for y in sorted(year_counts):
        bar = "█" * int(40 * year_counts[y] / max(year_counts.values()))
        print(f"  {y}  {year_counts[y]:5d}  {bar}")

    print(f"\nTop {top_n} terms overall")
    for w, c in overall.most_common(top_n):
        print(f"  {c:6d}  {w}")

    _print_trends(overall, by_year, year_counts)


def _share(by_year, year_counts, years):
    """Aggregate term share (per-1000 tokens) across a span of years."""
    c = Counter()
    n = 0
    for y in years:
        c.update(by_year[y])
        n += sum(by_year[y].values())
    if n == 0:
        return {}, 0
    return c, n


def _print_trends(overall, by_year, year_counts):
    years = sorted(year_counts)
    if len(years) < 4:
        print("\n(Not enough years for trend analysis.)")
        return

    mid = years[len(years) // 2]
    old_years = [y for y in years if y < mid]
    new_years = [y for y in years if y >= mid]

    old_c, old_n = _share(by_year, year_counts, old_years)
    new_c, new_n = _share(by_year, year_counts, new_years)
    if not old_n or not new_n:
        return

    def share(c, n, w):
        return 1000.0 * c.get(w, 0) / n

    candidates = [w for w in overall if overall[w] >= 50]
    deltas = sorted(
        ((share(new_c, new_n, w) - share(old_c, old_n, w), w) for w in candidates),
        reverse=True,
    )

    print(f"\nFastest-RISING terms  ({new_years[0]}+ vs <{mid}, per-1000 share)")
    for d, w in deltas[:20]:
        print(f"  +{d:5.2f}  {w}")

    print(f"\nFastest-FALLING terms")
    for d, w in deltas[-20:][::-1]:
        print(f"  {d:6.2f}  {w}")
