#!/usr/bin/env python3
"""Phase-1 pilot: run the four probe questions through baseline retrieval and
generation, and write reports/phase1_pilot.md.

One-off analysis script, not part of the library. Gold facts below were
established by term-diffing slide text against transcript text -- each
question's slide-only half is verified absent from the audio and vice versa.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from linkrag.core import load_config, setup_logging
from linkrag.generate.answer import answer, cited_ids
from linkrag.index import Index, default_encoder
from linkrag.retrieve.baseline import retrieve_scored

ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]?$")

# An answer that says "the evidence doesn't cover this" is behaving correctly on
# a retrieval failure -- materially different from one that invents the missing
# half. Distinguishing them is the difference between a retrieval bug and a
# generation bug, so it must not be collapsed into a single "partial".
REFUSAL = re.compile(
    r"(does not (state|provide|include|contain|specify)|not (in|stated in) the (provided |given )?"
    r"evidence|can.?t answer|cannot answer|no evidence|isn.?t (in|provided))", re.I)

# Hand-checked against the transcript and slide text after the 2026-09-06 run.
# The mechanical verdict above is term-matching only; it cannot see that an
# answer is confidently wrong, so these override it in the summary.
REVIEWED = {
    "Q1": ("partial (honest refusal; retrieval failure)",
           "Named NoScope correctly and explicitly declined the venue/year rather than "
           "inventing it. The failure is retrieval: slide p5 carries `Kang et al., NoScope, "
           "PVLDB'17` and was never returned -- 6 of 8 slots went to audio."),
    "Q2": ("complete",
           "Both halves correct and cited to hsieh:a52, the exact Q&A segment. The one "
           "question whose gold evidence is single-modality is the one the baseline gets right."),
    "Q3": ("WRONG (citation valid, answer false)",
           "Asserted 'Saurabh Bakshi from Purdue' as the named presenter, citing hsieh:a43. "
           "That string really is in a43 -- it is an *audience member introducing themselves "
           "to ask a question*, captured in the same unit as the talk's closing line. The "
           "citation check passes and the answer is still wrong. The title slide p1, which "
           "carries the real author list, was never retrieved. This is precisely the failure "
           "P2 cannot detect: it has no hallucination check, and a citation-validity test "
           "like ours passes here too."),
    "Q4": ("partial (honest refusal; retrieval failure)",
           "Recovered the speaker's spoken tradeoff options and correctly declined the plotted "
           "labels. All 8 retrieved units were audio -- slide p20, which carries `Optimize for "
           "Ingest Cost` / `Optimize for Query Latency`, never surfaced. A deictic phrase "
           "('in this figure', 'here we have') has no way to reach the figure it points at."),
}


@dataclass
class Probe:
    qid: str
    qtype: str
    question: str
    gold_modalities: set[str]
    # term -> which modality alone carries it. Presence in the ANSWER is what
    # decides completeness; presence in RETRIEVED text decides retrieval.
    gold_slide: list[str] = field(default_factory=list)
    gold_audio: list[str] = field(default_factory=list)


PROBES = [
    Probe("Q1", "slides_only",
          "Which prior system does the talk cite as the state-of-the-art query-time "
          "approach, and in which venue and year was it published?",
          {"text", "figure"},
          gold_slide=["kang", "pvldb"]),
    Probe("Q2", "audio_only",
          "In the Q&A, how does the speaker say the cheap CNNs are produced, and is "
          "that process automatic?",
          {"audio"},
          gold_audio=["resnet", "layer", "automatic"]),
    Probe("Q3", "cross_modal_split",
          "Who are the authors of Focus, and which institutions are they from?",
          {"text", "figure", "audio"},
          gold_slide=["ananthanarayanan", "bodik"],
          gold_audio=["microsoft", "carnegie"]),
    Probe("Q4", "cross_modal_deictic",
          "In the figure the speaker introduces by saying they plotted a set of "
          "different configurations, what are the configuration options labelled on "
          "the plot, and which one does the speaker say they would select?",
          {"text", "figure", "audio"},
          gold_slide=["optimize for ingest cost", "optimize for query latency"],
          gold_audio=["balance"]),
]


def coverage(found: set[str], wanted: set[str]) -> str:
    hit = found & wanted
    return "yes" if hit == wanted else ("partial" if hit else "no")


def segmentation_stats(units: list[dict]) -> dict:
    audio = [u for u in units if u["modality"] == "audio"]
    if not audio:
        return {}
    bad = [u for u in audio if not ENDS_SENTENCE.search(u["content"].strip())]
    spans = [u["location"]["end_s"] - u["location"]["start_s"] for u in audio]
    return {
        "n": len(audio),
        "bad": len(bad),
        "rate": 100 * len(bad) / len(audio),
        "mean_s": sum(spans) / len(spans),
        "max_s": max(spans),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default="reports/phase1_pilot.md")
    ap.add_argument("--baseline-index", default="data/processed/index_fixedwindow_backup",
                    help="unused; segmentation comparison reads COMPARISON_INDEXES")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    index = Index.load(args.index or cfg["index"]["store_dir"])
    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])  # warm outside the per-question timing

    live = Path(args.index or cfg["index"]["store_dir"])
    comparison = [
        ("fixed-window (baseline papers' approach)", Path("data/processed/index_fixedwindow_backup")),
        ("packed whisper segments (first attempt)", Path("data/processed/index_whisperseg_backup")),
        ("sentence-split + packed (shipped)", live),
    ]
    stats = []
    for label, path in comparison:
        f = path / "units.json"
        if f.exists():
            stats.append((label, segmentation_stats(json.loads(f.read_text()))))
    new = stats[-1][1] if stats else {}

    lines: list[str] = []
    w = lines.append
    w("# Phase 1 pilot — baseline multimodal RAG on `pilot01`")
    w("")
    w("Corpus: `hsieh.mp3` (22.9 min talk) + `osdi18_slides_hsieh.pdf` (27 slides) —")
    w("Hsieh et al., *Focus: Querying Large Video Datasets with Low Latency and Low Cost*, OSDI '18.")
    w("")
    w(f"Mode: **baseline** (RRF of dense + BM25, plain top-k, no links). ")
    w(f"Embedder `{index.embedding_model}`; LLM `{cfg['models']['llm']['model']}` "
      f"@ temperature {cfg['models']['llm']['temperature']}; top_k={cfg['retrieve']['top_k']}.")
    w("")

    w("## Segmentation")
    w("")
    w("Step 2 asked for sentence-aware segmentation *if* fixed windows cut mid-sentence.")
    w("They did (70%). The first fix -- packing whole whisper segments, as specified --")
    w("**did not work**: whisper's segments are decoder chunks, not sentences, so only")
    w("~26% of them end on punctuation and packing them left boundaries 74% mid-sentence,")
    w("marginally worse than doing nothing. Splitting the word stream on terminal")
    w("punctuation (whisper attaches it to words) and packing those sentences fixes it.")
    w("")
    w("| segmentation | audio units | boundaries mid-sentence | mean len | max len |")
    w("|---|---:|---:|---:|---:|")
    for label, st in stats:
        w(f"| {label} | {st['n']} | **{st['bad']}/{st['n']} ({st['rate']:.0f}%)** "
          f"| {st['mean_s']:.1f}s | {st['max_s']:.1f}s |")
    w("")
    w("Transcript content itself is good throughout: technical terms survive")
    w("(`ResNet 18`, `YOLO V2`, `NoScope`, `162 times`). Recurring ASR error: *ingest*")
    w("is transcribed as **\"interest\"** almost everywhere -- worth noting, since it")
    w("degrades BM25 recall on the single most important term in this talk.")
    w("")

    w("## Questions")
    w("")
    results = []
    for probe in PROBES:
        scored = retrieve_scored(probe.question, index, encoder=encoder,
                                 top_k=cfg["retrieve"]["top_k"],
                                 candidates=cfg["retrieve"]["candidates"],
                                 rrf_k=cfg["retrieve"]["rrf_k"])
        units = [u for u, _ in scored]
        text = answer(probe.question, units, cfg, mode="baseline")

        got_mod = {u.modality for u in units}
        retrieved_blob = " ".join(u.content for u in units).lower()
        answer_blob = text.lower()

        slide_in_retrieval = [t for t in probe.gold_slide if t in retrieved_blob]
        audio_in_retrieval = [t for t in probe.gold_audio if t in retrieved_blob]
        slide_in_answer = [t for t in probe.gold_slide if t in answer_blob]
        audio_in_answer = [t for t in probe.gold_audio if t in answer_blob]

        need_slide, need_audio = bool(probe.gold_slide), bool(probe.gold_audio)
        halves_ok = ((not need_slide or slide_in_answer) and (not need_audio or audio_in_answer))
        halves_retrieved = ((not need_slide or slide_in_retrieval) and
                            (not need_audio or audio_in_retrieval))
        declined = bool(REFUSAL.search(text))
        if halves_ok:
            verdict = "complete"
        elif slide_in_answer or audio_in_answer:
            verdict = "partial" + (" (explicit refusal on the missing half)" if declined else "")
        elif halves_retrieved:
            verdict = "wrong — evidence retrieved but unused"
        else:
            verdict = ("partial (explicit refusal; evidence never retrieved)" if declined
                       else "wrong — required evidence never retrieved")
        reviewed, reason = REVIEWED.get(probe.qid, (verdict, ""))

        w(f"### {probe.qid} · `{probe.qtype}`")
        w("")
        w(f"**Q.** {probe.question}")
        w("")
        w(f"**Gold modalities:** {', '.join(sorted(probe.gold_modalities))} — "
          f"retrieved modalities: {', '.join(sorted(got_mod))} → "
          f"**{coverage(got_mod, probe.gold_modalities)}**")
        w("")
        w("| rank | score | id | modality | location | content (truncated) |")
        w("|---:|---:|---|---|---|---|")
        for i, (u, s) in enumerate(scored, 1):
            snippet = " ".join(u.content.split())[:110].replace("|", "\\|")
            w(f"| {i} | {s:.4f} | `{u.id}` | {u.modality} | {u.location.cite()} | {snippet} |")
        w("")
        if probe.gold_slide:
            w(f"- slide-only gold terms {probe.gold_slide}: "
              f"in retrieval {slide_in_retrieval or 'NONE'}; in answer {slide_in_answer or 'NONE'}")
        if probe.gold_audio:
            w(f"- audio-only gold terms {probe.gold_audio}: "
              f"in retrieval {audio_in_retrieval or 'NONE'}; in answer {audio_in_answer or 'NONE'}")
        w("")
        w(f"**Answer.** {text.strip()}")
        w("")
        cited = cited_ids(text)
        unknown = [c for c in cited if c not in {u.id for u in units}]
        w(f"**Citations:** {len(cited)} of {len(units)} units"
          + (f" — **NOT IN EVIDENCE: {unknown}**" if unknown else ""))
        w("")
        w(f"**Verdict (mechanical): {verdict}**")
        w("")
        w(f"**Verdict (reviewed): {reviewed}** — {reason}")
        w("")
        results.append((probe, coverage(got_mod, probe.gold_modalities), reviewed, reason,
                        sorted(got_mod)))

    w("## Summary")
    w("")
    w("| Q | type | gold modalities retrieved | modalities returned | reviewed verdict |")
    w("|---|---|---|---|---|")
    for probe, cov, reviewed, _reason, mods in results:
        w(f"| {probe.qid} | `{probe.qtype}` | {cov} | {', '.join(mods)} | {reviewed} |")
    w("")
    w("## What the baseline actually failed at")
    w("")
    w("1. **Modality collapse.** 53 of 98 units are audio, and audio dominated every")
    w("   ranking: Q4 returned 8/8 audio, Q1 and Q3 returned 6/8. The slide carrying the")
    w("   answer was never retrieved for Q1, Q3 or Q4. Both rankers are modality-blind and")
    w("   RRF does nothing to rebalance them, so the larger modality wins on volume alone.")
    w("2. **A valid citation is not a correct answer.** Q3 cited a real unit and still")
    w("   asserted a false fact, because a Q&A self-introduction and an author list are")
    w("   indistinguishable to a flat index. Citation validity is necessary, not sufficient")
    w("   -- the eval needs a claim-level check, not just an id-level one.")
    w("3. **Deixis is unresolvable without links.** Q4's segment says \"in this figure\" and")
    w("   \"here we have\"; nothing in the baseline can turn either into a pointer to slide")
    w("   p20. This is the gap the Evidence Linking Layer exists to close.")
    w("4. **ASR noise hits the load-bearing term.** *ingest* is transcribed **\"interest\"**")
    w("   throughout, so BM25 cannot match the talk's central concept, and the dense side")
    w("   carries retrieval alone on exactly the queries that need it most.")
    w("")
    w("Honest caveat: the two questions the baseline handled acceptably (Q1's system name,")
    w("Q2 in full) are the ones whose evidence sits in a single modality. Every question")
    w("needing two modalities degraded. n=4 on one lecture -- indicative, not a result.")
    w("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for probe, cov, reviewed, _reason, _mods in results:
        print(f"  {probe.qid} {probe.qtype:22} gold={cov:8} {reviewed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
