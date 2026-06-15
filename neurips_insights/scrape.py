"""Scrape NeurIPS paper metadata (title + abstract + authors) — no PDFs.

Streams each record to a JSONL file immediately, so the run is fully resumable
and uses constant memory regardless of corpus size.
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

from .config import BASE, HEADERS, DELAY_MIN, DELAY_MAX
from .corpus import append_record


def _abs_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return BASE + (href if href.startswith("/") else "/" + href)


def get_paper_links_for_year(year: int) -> List[str]:
    """Return deduped list of per-paper Abstract page URLs for a given year."""
    url = f"{BASE}/paper_files/paper/{year}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ! failed to fetch index for {year}: {e}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    links = [
        _abs_url(a["href"])
        for a in soup.find_all("a", href=True)
        if "hash" in a["href"] and "Abstract" in a["href"]
    ]
    # dedupe, preserve order
    return list(dict.fromkeys(links))


def _extract_abstract(soup: BeautifulSoup) -> str:
    """Prefer citation_abstract meta; fall back to the longest paragraph."""
    meta = soup.find("meta", attrs={"name": "citation_abstract"})
    if meta and meta.get("content") and len(meta["content"].split()) > 10:
        return meta["content"].strip()

    # Fallback: NeurIPS renders the abstract as the longest <p> on the page.
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paras = [p for p in paras if len(p.split()) > 15]  # drop nav/footer boilerplate
    return max(paras, key=len) if paras else ""


def extract_paper(url: str) -> Optional[Dict[str, Any]]:
    """Fetch one paper page and return its metadata dict (no files downloaded)."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    meta_t = soup.find("meta", attrs={"name": "citation_title"})
    if meta_t and meta_t.get("content"):
        title = meta_t["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.split("|")[0].strip()
    else:
        title = ""

    authors = [
        m["content"]
        for m in soup.find_all("meta", attrs={"name": "citation_author"})
        if m.get("content")
    ]

    abstract = _extract_abstract(soup)

    return {"title": title, "abstract": abstract, "authors": authors, "url": url}


def _load_done(log_path: str) -> set:
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def scrape(
    start_year: int,
    end_year: int,
    corpus_path: str,
    log_path: str,
    limit_per_year: Optional[int] = None,
) -> None:
    """Scrape all papers in [start_year, end_year] into corpus_path (JSONL).

    Resumable: a per-paper hash log skips already-scraped papers.
    """
    done = _load_done(log_path)
    print(f"Resuming: {len(done)} papers already scraped.")

    for year in range(start_year, end_year + 1):
        links = get_paper_links_for_year(year)
        if limit_per_year:
            links = links[:limit_per_year]
        print(f"\nNeurIPS {year}: {len(links)} papers")

        new_count = 0
        for i, url in enumerate(links, 1):
            h = url.rstrip("/").split("/")[-1]
            if h in done:
                continue
            try:
                rec = extract_paper(url)
                if rec is None:
                    continue
                rec["year"] = year
                append_record(corpus_path, rec)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(h + "\n")
                    lf.flush()
                done.add(h)
                new_count += 1
                if new_count % 50 == 0:
                    print(f"  [{i}/{len(links)}] {year} (+{new_count} new)")
            except Exception as e:
                print(f"  ! err {h}: {e}")
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        print(f"Done {year}: +{new_count} new papers.")

    print("\nScrape complete.")
