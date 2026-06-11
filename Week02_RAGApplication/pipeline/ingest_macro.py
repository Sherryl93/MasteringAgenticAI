"""Macro corpus ingestion.

Fetches three macro source families into LangChain Documents:
  - FOMC minutes  : PDF  via PyMuPDFLoader
  - BLS CPI       : HTML via WebBaseLoader (p/table/h2/h3 only)
  - BLS NFP       : HTML via WebBaseLoader (p/table/h2/h3 only)

Every returned Document carries `source_type` and `url` metadata so the
downstream retrieval/synthesis layers can cite the originating report.
"""

from __future__ import annotations

import bs4
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_core.documents import Document

from pipeline import config


def _load_fomc() -> list[Document]:
    """Load FOMC minutes PDFs and tag them as macro/fomc."""
    docs: list[Document] = []
    for url in config.FOMC_URLS:
        try:
            loaded = PyMuPDFLoader(url).load()
        except Exception as exc:  # noqa: BLE001 — keep ingesting remaining URLs
            print(f"[macro] WARN  FOMC fetch failed: {url} -> {exc}")
            continue
        for d in loaded:
            d.metadata["source_type"] = config.SRC_FOMC
            d.metadata["url"] = url
        docs.extend(loaded)
        print(f"[macro] OK    FOMC {url} ({len(loaded)} pages)")
    return docs


def _load_bls_html(urls: list[str], source_type: str) -> list[Document]:
    """Load BLS HTML releases, parsing only p/table/h2/h3 tags."""
    strainer = bs4.SoupStrainer(config.BLS_PARSE_TAGS)
    docs: list[Document] = []
    for url in urls:
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={"parse_only": strainer},
            )
            loaded = loader.load()
        except Exception as exc:  # noqa: BLE001
            print(f"[macro] WARN  {source_type.upper()} fetch failed: {url} -> {exc}")
            continue
        for d in loaded:
            d.metadata["source_type"] = source_type
            d.metadata["url"] = url
        docs.extend(loaded)
        print(f"[macro] OK    {source_type.upper()} {url} ({len(loaded)} docs)")
    return docs


def load_macro_documents() -> list[Document]:
    """Fetch the full macro corpus (FOMC + CPI + NFP) as tagged Documents."""
    docs: list[Document] = []
    docs.extend(_load_fomc())
    docs.extend(_load_bls_html(config.BLS_CPI_URLS, config.SRC_CPI))
    docs.extend(_load_bls_html(config.BLS_NFP_URLS, config.SRC_NFP))
    print(f"[macro] TOTAL {len(docs)} macro documents loaded")
    return docs


if __name__ == "__main__":
    loaded = load_macro_documents()
    print(f"\nLoaded {len(loaded)} macro documents.")
    if loaded:
        sample = loaded[0]
        print("Sample metadata:", sample.metadata)
        print("Sample text (first 300 chars):", sample.page_content[:300])
