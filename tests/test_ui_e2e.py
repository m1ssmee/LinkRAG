"""Lectern end to end, in a real browser: the server on the frozen pilot01 corpus (read, never
written), real bge-m3 query embeddings, and the mock answerer and judge -- nothing bills. Walks
the empty, answer, compare and abstention states and saves their screenshots to reports/ui/.

    LECTERN_SAMPLE_DIR=/path/to/data/processed pytest -m slow tests/test_ui_e2e.py

Needs the built frontend (src/linkrag/ui/web: npm ci && npm run build) and Chromium
(`playwright install chromium`). The first sample load computes its file-pair gates in the
background (the document-pair gate re-embeds shuffled text), which takes minutes on a CPU; set
LECTERN_WORKDIR to a kept directory and a rerun reads them from its cache.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(os.environ.get("LECTERN_SAMPLE_DIR", ROOT / "data" / "processed"))
DIST = ROOT / "src" / "linkrag" / "ui" / "web" / "dist" / "index.html"
SHOTS = ROOT / "reports" / "ui"
QUESTION = "How does Focus use clustering to reduce query cost?"
NOT_IN_IT = "What is the recipe for sourdough bread?"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (SAMPLE / "index" / "units.json").exists(),
                       reason=f"no frozen corpus at {SAMPLE} (set LECTERN_SAMPLE_DIR)"),
    pytest.mark.skipif(not DIST.exists(), reason="frontend not built: cd src/linkrag/ui/web && npm run build"),
]


def _files_read() -> list[tuple[str, int, int]]:
    """What the UI reads from the frozen corpus: the index, links, manifest, raw pilot01 media."""
    paths = [*(SAMPLE / "index").iterdir(), SAMPLE / "links.jsonl", SAMPLE / "manifest.json",
             *(SAMPLE.parent / "raw" / "pilot01").glob("*")]
    return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file())


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    pytest.importorskip("playwright.sync_api")
    import uvicorn

    from linkrag.core import load_config
    from linkrag.ui import api

    workdir = os.environ.get("LECTERN_WORKDIR") or tmp_path_factory.mktemp("lectern")
    app = api.create_app(load_config(ROOT / "configs" / "default.yaml"), workdir=Path(workdir),
                         sample=SAMPLE, sample_title="Focus: Querying Large Video Datasets (OSDI ’18)",
                         answerer="mock", ledger=None)
    threading.Thread(target=app.state.preload, daemon=True).start()          # as main() does
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
    thread.join(timeout=10)


def _settle(page) -> None:
    """Page renders are made on first request: wait for them, then drop focus and the pointer
    so a screenshot shows the state, not the last interaction."""
    page.wait_for_function("() => [...document.querySelectorAll('aside img')].every(i => i.complete)",
                           timeout=180_000)
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(700, 600)
    page.wait_for_timeout(400)


def _ask(page, question: str) -> None:
    """Ask, then wait for this question's answer: the question shows at once, the loading line
    until the answer lands (the previous answer's claims must not count)."""
    box = page.get_by_role("textbox", name="Question")
    box.fill(question)
    box.press("Enter")
    page.locator("#question-label + p").get_by_text(question, exact=True).wait_for()
    page.get_by_role("status").wait_for(state="detached", timeout=300_000)


def test_the_states_in_a_real_browser(server):
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    before = _files_read()
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(server)

        # 1: the drop zone and the sample, nothing else
        sample = page.get_by_role("button", name="Try the sample lecture")
        sample.wait_for()
        assert page.get_by_text("Drop lecture files here").is_visible()
        page.screenshot(path=SHOTS / "state1_empty.png")

        sample.click()
        page.get_by_role("heading", name="Materials").wait_for(timeout=900_000)
        # the connectors show the gate's z once the sample's gates land (computed in the background)
        page.wait_for_function("() => !document.querySelector('aside').innerText.includes('checking')",
                               timeout=1_800_000)
        assert page.get_by_text("linked · ", exact=False).count() >= 1

        # 3: an answer -- verified claims citing speech, slides and notes; a speech citation
        # seeks the recording, so the filmstrip and playhead move with it
        _ask(page, QUESTION)
        claims = page.locator("[data-claim]")
        assert claims.count() >= 2
        assert page.get_by_text("reached through links", exact=False).is_visible()
        assert page.get_by_text("every verdict holds by construction", exact=False).is_visible()   # mock, labelled
        speech_chip = page.locator("[data-claim] button.text-speech").first
        speech_chip.click()
        assert not page.locator("footer .tabular-nums").inner_text().startswith("0:00 /")   # the playhead moved
        _settle(page)
        claims.nth(1).hover()
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "state3_answer.png")

        # 6: why this evidence -- the sub-graph as a modal sheet, closed with Escape
        page.get_by_role("button", name="Why this evidence?").click()
        sheet = page.get_by_role("dialog", name="Why this evidence?")
        sheet.get_by_text("Links among this evidence").wait_for(timeout=60_000)
        assert sheet.locator("svg circle").count() >= 2
        page.keyboard.press("Escape")
        sheet.wait_for(state="detached")
        assert page.evaluate("document.activeElement.id") == "why-evidence"

        # 4: the same question beside the baseline; evidence it did not retrieve is dimmed and tagged
        page.get_by_role("switch", name="Compare with baseline").click()
        page.get_by_role("region", name="Baseline").wait_for(timeout=300_000)
        assert page.get_by_role("region", name="Lectern").is_visible()
        unretrieved = page.get_by_text("not retrieved by baseline")
        assert unretrieved.count() >= 1
        _settle(page)
        unretrieved.first.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=SHOTS / "state4_compare.png")

        # 5: nothing in the material answers: one grey line, no claims
        page.get_by_role("switch", name="Compare with baseline").click()
        page.get_by_role("region", name="Baseline").wait_for(state="detached", timeout=300_000)
        page.get_by_role("status").wait_for(state="detached", timeout=300_000)   # switching re-asks
        _ask(page, NOT_IN_IT)
        assert page.get_by_text("Not found in the provided material").is_visible()
        assert page.locator("[data-claim]").count() == 0
        _settle(page)
        page.screenshot(path=SHOTS / "state5_abstention.png")
        browser.close()

    assert errors == []
    assert _files_read() == before              # the frozen corpus was read, never written
