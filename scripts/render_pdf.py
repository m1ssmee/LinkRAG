#!/usr/bin/env python3
"""Print an HTML report, or a Markdown file, to an A4 PDF with Playwright's Chromium (requirements-ui.txt).

    python scripts/render_pdf.py reports/pdf/results_detailed.html reports/LinkRAG_results_detailed.pdf
    uv run --with markdown python scripts/render_pdf.py reports/study_guide/study_guide.md reports/LinkRAG_study_guide.pdf

Markdown is converted with the `markdown` package and styled with reports/pdf/report.css.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FOOTER = ('<div style="font: 7pt Helvetica, Arial, sans-serif; color: #5b6270; width: 100%; text-align: center;">'
          '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')


def markdown_page(src: Path) -> Path:
    """The Markdown as a standalone HTML page next to it, so its relative image paths resolve."""
    import markdown
    css = (Path(__file__).resolve().parents[1] / "reports" / "pdf" / "report.css").read_text()
    body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code", "sane_lists"])
    page = src.with_name(f".{src.stem}.render.html")
    page.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{src.stem}</title>'
                    f'<style>{css} img {{ max-width: 100%; }} blockquote {{ margin: 2mm 0; padding-left: 4mm; '
                    f'border-left: 2px solid var(--rule); color: var(--muted); }}</style></head><body>{body}</body></html>')
    return page


def render(src: Path, out: Path) -> None:
    page = markdown_page(src) if src.suffix == ".md" else None
    try:
        _render(page or src, out)
    finally:
        if page:
            page.unlink()


def _render(src: Path, out: Path) -> None:
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:           # Playwright's own Chromium is not downloaded: use the installed Chrome
            browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(src.resolve().as_uri(), wait_until="load")
        page.pdf(path=str(out), prefer_css_page_size=True, print_background=True,
                 display_header_footer=True, header_template="<span></span>", footer_template=FOOTER)
        browser.close()


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]))
