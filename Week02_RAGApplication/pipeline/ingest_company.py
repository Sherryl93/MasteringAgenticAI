"""Company corpus ingestion — NVDA SEC filings.

Uses sec-edgar-downloader to fetch:
  - 10-K : most recent annual report
  - 8-K  : recent filings, filtered to Item 2.02 (Results of Operations /
           earnings releases), keeping the last 4 quarters.

Raw EDGAR submissions are SGML/HTML/XBRL soup, so each filing runs through a
fixed preprocessing pipeline before becoming a LangChain Document:
  1. html.unescape()                       — decode HTML entities
  2. strip SGML/HTML tags                   — re.sub(r'<[^>]+>', ' ', text)
  3. strip XBRL inline fragments            — re.sub(r'ix:[^\\s]+', ' ', text)
  4. drop EDGAR header before first         — "UNITED STATES SECURITIES"
  5. collapse excessive whitespace / blanks
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from pathlib import Path

from langchain_core.documents import Document
from sec_edgar_downloader import Downloader

from pipeline import config

# SEC fair-access requires a declared requester identity. These are only used
# in the User-Agent of EDGAR requests — not secrets, safe to keep in code.
EDGAR_REQUESTER_NAME = "Macro-Micro Bridge Agent"
EDGAR_REQUESTER_EMAIL = "research@example.com"

# Known CIKs for the supported tickers — the fast path, no network call.
# Any ticker not listed here is resolved live via SEC (see _resolve_cik).
# Downloads use the ticker symbol directly, so a missing CIK only leaves the
# accession URL metadata blank; it never breaks ingestion.
TICKER_CIK = {
    "NVDA": "1045810",
    "AAPL": "320193",
    "MSFT": "789019",
    "AMD": "2488",
}

# Official SEC ticker -> CIK directory (covers every public filer).
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Per-process cache of the full ticker -> CIK map (lazily fetched once).
_cik_map: dict[str, str] | None = None

# Where sec-edgar-downloader writes filings (matches .gitignore entry).
DOWNLOAD_ROOT = Path("sec-edgar-filings")

# How many recent 8-Ks to pull before filtering down to Item 2.02 earnings.
MAX_8K_SCAN = 20
MAX_EARNINGS_QUARTERS = 4

_ITEM_202_PATTERNS = [
    re.compile(r"item\s*2\.02", re.IGNORECASE),
    re.compile(r"results\s+of\s+operations", re.IGNORECASE),
]


def clean_edgar_text(raw: str) -> str:
    """Clean a raw EDGAR submission into readable prose, stripping XBRL noise."""
    # 1. decode HTML entities
    text = html.unescape(raw)
    # 2. strip SGML / HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # 3. strip XBRL inline fragments (e.g. ix:nonNumeric)
    text = re.sub(r"ix:[^\s]+", " ", text)
    # 4. drop EDGAR header — keep from first "UNITED STATES SECURITIES" onward
    marker = text.find("UNITED STATES SECURITIES")
    if marker != -1:
        text = text[marker:]

    # 5. Remove XBRL element name patterns (namespace:TypeName, camelCase)
    text = re.sub(r"\b[a-z]+:[A-Za-z]+\b", " ", text)

    # 6. Remove lines that are only XBRL-style identifiers
    #    e.g. "nvda_WarrantyLiabilityTermOfWarranties"
    text = re.sub(
        r"^[a-z]{2,10}_[A-Z][A-Za-z]{10,}\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 7. Remove lines containing only metadata labels
    #    e.g. "Balance Type: na", "Period Type: duration", "Data Type: xbrli:..."
    text = re.sub(
        r"^(Balance Type|Period Type|Data Type|Namespace Prefix)\s*:.*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 8. Remove lines with only a single character (XBRL placeholders)
    text = re.sub(r"^\s*[a-zA-Z]\s*$", "", text, flags=re.MULTILINE)

    # 8b. Remove EDGAR R-file index / table-of-contents entries, e.g.
    #     "false false R84.htm 9955564 - Disclosure - Segment Information ...".
    text = re.sub(
        r"(?:true|false)\s+(?:true|false)\s+R\d+\.htm[^\n]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 8c. Strip recurring legal boilerplate *in place* (keep the surrounding real
    #     content — e.g. the income statement that follows the disclaimer).
    for phrase in ("subject to change without notice", "all rights reserved"):
        text = re.sub(
            rf"[^.\n]*\b{phrase}\b[^.\n]*\.?", " ", text, flags=re.IGNORECASE
        )

    # 9. collapse excessive whitespace and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _load_sec_cik_map() -> dict[str, str]:
    """Fetch and cache SEC's full ticker -> CIK directory (once per process)."""
    global _cik_map
    if _cik_map is not None:
        return _cik_map
    try:
        req = urllib.request.Request(
            SEC_TICKER_MAP_URL,
            headers={"User-Agent": f"{EDGAR_REQUESTER_NAME} {EDGAR_REQUESTER_EMAIL}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.load(resp)
        # Shape: {"0": {"cik_str": 1045810, "ticker": "NVDA", ...}, ...}
        _cik_map = {
            entry["ticker"].upper(): str(entry["cik_str"])
            for entry in raw.values()
        }
        print(f"[company] loaded SEC ticker->CIK map ({len(_cik_map)} symbols)")
    except Exception as exc:  # noqa: BLE001 — fall back to the offline map
        print(f"[company] WARN  SEC CIK lookup failed ({exc}); using fallback map")
        _cik_map = {}
    return _cik_map


def _resolve_cik(ticker: str) -> str:
    """Resolve a ticker's CIK.

    Fast path: the local TICKER_CIK dict (no network) for known tickers.
    Fallback: a live SEC lookup, used only when the ticker isn't in the dict.
    """
    ticker = ticker.upper()
    if ticker in TICKER_CIK:
        return TICKER_CIK[ticker]
    return _load_sec_cik_map().get(ticker, "")


def _accession_url(accession_dir: str, ticker: str) -> str:
    """Best-effort EDGAR filing-index URL from an accession folder name."""
    cik = _resolve_cik(ticker)
    if not cik:
        return ""
    accession_nodash = accession_dir.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_nodash}/"
    )


def _filing_dirs(form: str, ticker: str) -> list[Path]:
    """Return downloaded accession directories for a given form, newest first.

    sec-edgar-downloader names accession folders with the accession number,
    which sorts chronologically, so reverse-sorted == newest first.
    """
    base = DOWNLOAD_ROOT / ticker / form
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def _read_submission(accession_dir: Path) -> str | None:
    """Read the full-submission text file from an accession directory."""
    candidate = accession_dir / "full-submission.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8", errors="ignore")
    # Fallback: any .txt the downloader produced.
    for txt in accession_dir.glob("*.txt"):
        return txt.read_text(encoding="utf-8", errors="ignore")
    return None


def _read_primary_document(accession_dir: Path) -> str | None:
    """Prefer the primary filing document (the narrative report) over the full
    submission. The 10-K's ``primary-document.html`` is the actual report; the
    full submission also concatenates the bulky XBRL exhibits (~6x larger, mostly
    machine metadata), so the primary document gives far cleaner narrative.
    """
    primary = accession_dir / "primary-document.html"
    if primary.exists():
        return primary.read_text(encoding="utf-8", errors="ignore")
    return _read_submission(accession_dir)


def _is_item_202(text: str) -> bool:
    """True if the filing looks like an Item 2.02 earnings release."""
    return any(p.search(text) for p in _ITEM_202_PATTERNS)


def _download_filings(ticker: str) -> None:
    """Fetch a ticker's 10-K (1) and recent 8-Ks via sec-edgar-downloader."""
    dl = Downloader(
        EDGAR_REQUESTER_NAME,
        EDGAR_REQUESTER_EMAIL,
        str(DOWNLOAD_ROOT.parent if DOWNLOAD_ROOT.parent != Path(".") else "."),
    )
    print(f"[company] downloading most recent {ticker} 10-K ...")
    dl.get("10-K", ticker, limit=1, download_details=True)
    print(f"[company] downloading up to {MAX_8K_SCAN} recent {ticker} 8-Ks ...")
    dl.get("8-K", ticker, limit=MAX_8K_SCAN, download_details=True)


def _build_10k_documents(ticker: str) -> list[Document]:
    """Load and clean the most recent 10-K into a single Document."""
    docs: list[Document] = []
    dirs = _filing_dirs("10-K", ticker)
    if not dirs:
        print(f"[company] WARN  no {ticker} 10-K filings downloaded")
        return docs
    accession_dir = dirs[0]
    raw = _read_primary_document(accession_dir)
    if not raw:
        print(f"[company] WARN  no submission text in {accession_dir.name}")
        return docs
    cleaned = clean_edgar_text(raw)
    docs.append(
        Document(
            page_content=cleaned,
            metadata={
                "source_type": config.SRC_10K,
                "ticker": ticker,
                "url": _accession_url(accession_dir.name, ticker),
            },
        )
    )
    print(f"[company] OK    {ticker} 10-K {accession_dir.name} ({len(cleaned)} chars)")
    return docs


def _build_earnings_documents(ticker: str) -> list[Document]:
    """Load recent 8-Ks, keep Item 2.02 earnings releases (last 4 quarters)."""
    docs: list[Document] = []
    for accession_dir in _filing_dirs("8-K", ticker):
        if len(docs) >= MAX_EARNINGS_QUARTERS:
            break
        raw = _read_submission(accession_dir)
        if not raw:
            continue
        if not _is_item_202(raw):
            continue
        cleaned = clean_edgar_text(raw)
        docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source_type": config.SRC_EARNINGS,
                    "ticker": ticker,
                    "url": _accession_url(accession_dir.name, ticker),
                },
            )
        )
        print(
            f"[company] OK    {ticker} 8-K (Item 2.02) {accession_dir.name} "
            f"({len(cleaned)} chars)"
        )
    if not docs:
        print(f"[company] WARN  no Item 2.02 earnings 8-Ks found for {ticker}")
    return docs


def ingest_company(ticker: str, download: bool = True) -> list[Document]:
    """Fetch + clean one ticker's company corpus (10-K + earnings 8-Ks).

    Validates the ticker against the supported whitelist before ingesting.
    """
    ticker = config.validate_ticker(ticker)
    if download:
        _download_filings(ticker)
    docs: list[Document] = []
    docs.extend(_build_10k_documents(ticker))
    docs.extend(_build_earnings_documents(ticker))
    print(f"[company] TOTAL {len(docs)} {ticker} company documents loaded")
    return docs


# Backwards-compatible alias.
def load_company_documents(ticker: str = config.TICKER, download: bool = True) -> list[Document]:
    """Alias for :func:`ingest_company` (kept for existing call sites)."""
    return ingest_company(ticker, download=download)


if __name__ == "__main__":
    loaded = ingest_company(config.TICKER)
    print(f"\nLoaded {len(loaded)} company documents.")
    for d in loaded:
        print(
            f"- {d.metadata['source_type']:<20} "
            f"{len(d.page_content):>8} chars  url={d.metadata.get('url')}"
        )
