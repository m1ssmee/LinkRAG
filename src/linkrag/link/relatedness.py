"""Relatedness gate -- priority (ii) in DESIGN.md.

Every link the Evidence Linking Layer emits says "these two units belong together".
The alignment, layout and deixis signals that produce links are *structural*: a
segment was spoken while a slide was up, a figure sits above a paragraph, a pronoun
fired on an aligned slide. Structure is evidence of relatedness, not proof of it --
a lecturer digresses, a figure sits above an unrelated footnote, "this" points at
the demo and not the slide.

The gate asks the judge, per link: are these two units about the same specific
content? `runs` runs at temperature 0, majority vote, plus a short phrase naming the
shared subject so a pass is checkable. The verdict is written into the link's
`metadata["relatedness"]`; `load_links(..., gated=True)` (the default once a file
has been gated) drops failed links before the graph is built, and the per-type pass
rate is reported. A link type whose pass rate is low is a link type whose *signal*
is weak on this corpus -- that is the number to read, not just the filtered graph.

Applied to every link type, `same_slide` included: co-location on a deck page is
certain, but a decorative figure and the slide's text are still not "about the same
thing", and the reranker's beta term should not be paid for that edge.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from linkrag.core import EvidenceUnit, Link
from linkrag.eval.verify_gold import JUDGE_SYSTEM, majority, parse_json

Completer = Callable[[str, str], str]

PROMPT = """Two pieces of evidence from the same lecture corpus.

A ({a_kind}, {a_where}):
\"\"\"{a_text}\"\"\"

B ({b_kind}, {b_where}):
\"\"\"{b_text}\"\"\"

Are A and B about the SAME specific content -- does one state, explain, illustrate,
or refer to what the other says? "Same lecture" or "same general topic" is not
enough; a shared named thing (a system, a number, a plot, a claim) is. OCR noise in
a figure is not a reason to say no if its readable words match.

Output JSON only: {{"related": true|false, "subject": "<3-8 words naming the shared
content, or empty>"}}"""

KIND = {"audio": "spoken segment", "text": "text", "figure": "figure (OCR / caption)",
        "table": "table"}


@dataclass
class LinkVerdict:
    src_id: str
    dst_id: str
    link_type: str
    passed: bool
    votes: list[str]
    subject: str


def _where(u: EvidenceUnit) -> str:
    from pathlib import Path
    name = Path(u.source_file).name
    loc = u.location
    if loc.page is not None:
        return f"{name} p.{loc.page}"
    if loc.start_s is not None:
        return f"{name} {int(loc.start_s // 60)}:{int(loc.start_s % 60):02d}"
    return name


def judge_link(link: Link, a: EvidenceUnit, b: EvidenceUnit, judge: Completer, runs: int,
               max_chars: int = 2500) -> LinkVerdict:
    prompt = PROMPT.format(a_kind=KIND.get(a.modality, a.modality), a_where=_where(a),
                           a_text=" ".join(a.content.split())[:max_chars],
                           b_kind=KIND.get(b.modality, b.modality), b_where=_where(b),
                           b_text=" ".join(b.content.split())[:max_chars])
    votes, subjects = [], []
    for _ in range(runs):
        reply = parse_json(judge(JUDGE_SYSTEM, prompt))
        ok = bool(reply.get("related", False))
        votes.append("yes" if ok else "no")
        if ok:
            subjects.append(str(reply.get("subject", "") or ""))
    return LinkVerdict(link.src_id, link.dst_id, link.link_type, majority(votes, "yes"), votes,
                       subjects[0] if subjects else "")


def gate_links(links: Sequence[Link], units: Sequence[EvidenceUnit], judge: Completer, *,
               runs: int = 3, workers: int = 6,
               progress: Callable[[str], None] | None = None) -> list[LinkVerdict]:
    by_id = {u.id: u for u in units}
    todo = [l for l in links if l.src_id in by_id and l.dst_id in by_id]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(judge_link, l, by_id[l.src_id], by_id[l.dst_id], judge, runs)
                   for l in todo]
        verdicts = [f.result() for f in futures]
    if progress:
        for t, (n, k) in pass_rates(verdicts).items():
            progress(f"{t:<12} {k}/{n} = {k / n:.1%} pass")
    return verdicts


def apply_verdicts(links: Sequence[Link], verdicts: Sequence[LinkVerdict]) -> list[Link]:
    """Write each verdict into its link's metadata (links are returned, not filtered)."""
    key = {(v.src_id, v.dst_id, v.link_type): v for v in verdicts}
    out = []
    for l in links:
        v = key.get((l.src_id, l.dst_id, l.link_type))
        if v is not None:
            l.metadata = {**(l.metadata or {}),
                          "relatedness": {"passed": v.passed, "votes": v.votes, "subject": v.subject}}
        out.append(l)
    return out


def pass_rates(verdicts: Sequence[LinkVerdict]) -> dict[str, tuple[int, int]]:
    n, k = Counter(), Counter()
    for v in verdicts:
        n[v.link_type] += 1
        k[v.link_type] += int(v.passed)
    return {t: (n[t], k[t]) for t in sorted(n)}


def is_gated_out(link: Link) -> bool:
    r = (link.metadata or {}).get("relatedness")
    return bool(r) and not r.get("passed", True)


def write_report(verdicts: Sequence[LinkVerdict], links: Sequence[Link],
                 units: Sequence[EvidenceUnit], path, *, corpus: str, manifest_hash: str,
                 judge_model: str, runs: int, sample: int = 6) -> Any:
    from pathlib import Path
    by_id = {u.id: u for u in units}
    rates = pass_rates(verdicts)
    total_n = sum(n for n, _ in rates.values())
    total_k = sum(k for _, k in rates.values())
    L = [f"# Relatedness gate — {corpus}", "",
         f"Corpus `{manifest_hash}` · {len(links)} links judged · judge `{judge_model}` · "
         f"temperature 0 · {runs} runs, majority · generated by `linkrag.link.relatedness`", "",
         "**Reading it.** A link passes when the judge says its two units are about the same "
         "specific content. The pass rate per type is the strength of that type's *signal* "
         "on this corpus; failed links stay in `links.jsonl` flagged and are dropped when the "
         "graph is built (`load_links(gated=True)`).", "",
         "| link type | judged | passed | pass rate |", "|---|---:|---:|---:|"]
    for t, (n, k) in rates.items():
        L.append(f"| {t} | {n} | {k} | **{k / n:.1%}** |")
    L.append(f"| **all** | {total_n} | {total_k} | **{total_k / total_n:.1%}** |")
    L += ["", f"Per-type counts sum to the total (measurement rule 2): {total_n} = {len(verdicts)}.", ""]

    def short(uid: str, n: int = 110) -> str:
        u = by_id.get(uid)
        return "" if u is None else " ".join(u.content.split())[:n]

    for t in rates:
        fails = [v for v in verdicts if v.link_type == t and not v.passed]
        passes = [v for v in verdicts if v.link_type == t and v.passed]
        L += [f"## {t}", "", f"**Failed ({len(fails)}), sample:**", ""]
        for v in fails[:sample]:
            L.append(f"- `{v.src_id}` → `{v.dst_id}` ({'/'.join(v.votes)})  \n"
                     f"  A: {short(v.src_id)}  \n  B: {short(v.dst_id)}")
        L += ["", f"**Passed ({len(passes)}), sample with the judge's subject:**", ""]
        for v in passes[:sample]:
            L.append(f"- `{v.src_id}` → `{v.dst_id}` — *{v.subject}*")
        L.append("")
    path = Path(path)
    path.write_text("\n".join(L) + "\n")
    return path
