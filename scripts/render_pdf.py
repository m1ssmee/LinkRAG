#!/usr/bin/env python3
"""Print an HTML report to an A4 PDF with Playwright's Chromium (requirements-ui.txt).

    python scripts/render_pdf.py reports/pdf/results_detailed.html reports/LinkRAG_results_detailed.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FOOTER = ('<div style="font: 7pt Helvetica, Arial, sans-serif; color: #5b6270; width: 100%; text-align: center;">'
          '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')


def render(src: Path, out: Path) -> None:
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
