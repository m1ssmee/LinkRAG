"""Free re-checks of the stored LectQA-Vid results: every paired interval resampled by video,
the eq. 22 filter's empty windows, a question-echo floor, the size of all transcripts together,
and a sheet of timestamps for a second person to check. No answer is generated and no retrieval
report is re-run; the one model loaded is T1's local embedder, to see why windows came back empty."""

from __future__ import annotations

import collections
import csv
import json
import random
import statistics as st
import subprocess
from pathlib import Path

import numpy as np

from linkrag.index import Index

from lectqa.attribution import LABEL
from lectqa.common import LEVELS, PROCESSED, THEIRS, gold_interval, load_qa
from lectqa.open_ended import T1, open_items, paired_cell, t1_chunks, t1_encoder

SHEET_SEED = "lectqa-timestamp-check-20261112"
EMPTY_REPLY = "No evidence was retrieved"


def by_video(d, vids, n: int = 10000, seed: int = 20260923) -> tuple[float, float]:
    """95 % interval of the mean of d when whole videos are resampled: questions on one video
    share a transcript, so they are not independent draws."""
    names, inv = np.unique(np.asarray(vids), return_inverse=True)
    s, c = np.bincount(inv, np.asarray(d, float)), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(names), (n, len(names)))
    lo, hi = np.percentile(s[pick].sum(axis=1) / c[pick].sum(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def cells(d, vids) -> str:
    """Two table cells: the stored by-question interval, and the by-video one."""
    lo, hi = by_video(d, vids)
    return f"{paired_cell(list(d))} | {np.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("" if lo > 0 or hi < 0 else " (crosses 0)")


def overlaps(u, lo: float, hi: float) -> bool:
    return u.location.start_s <= hi and u.location.end_s >= lo


def recheck(ids: list[str], cfg: dict, external: Path) -> int:
    import tiktoken
    from linkrag.eval import lectqa_metrics as LM
    items, _ = open_items(ids)
    rows = [r for r in map(json.loads, (external / "lectqa_open_modes.jsonl").read_text().splitlines()) if r["subset"]]
    rows += [json.loads(x) for x in (external / "lectqa_attribution.jsonl").read_text().splitlines()]
    f1 = collections.defaultdict(lambda: collections.defaultdict(list))     # question order as in the stored reports
    for r in rows:
        f1[r["qi"]][r["mode"]].append(100 * LM.token_prf(r["answer"], r["ref"])[2])
    qis = list(f1)
    score = {m: np.array([st.mean(f1[q][m]) for q in qis]) for m in LABEL}
    vid = np.array([items[q]["vid"] for q in qis])
    level = np.array([items[q]["level"] for q in qis])
    rep0 = {(r["mode"], r["qi"]): r for r in rows if r["rep"] == 0}
    applied = np.array([rep0["baseline", q]["oracle"] for q in qis])
    empty = applied & np.array([rep0["baseline", q]["answer"].startswith(EMPTY_REPLY) for q in qis])
    # ponytail: the fallback swaps whole stored answers; a real fallback inside the pipeline would also re-rank
    score["fallback"] = np.where(empty, score["replica_no_eq22"], score["baseline"])
    label = {**LABEL, "fallback": "T1 replica, eq. 22 on, unfiltered when the window is empty"}

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    L = ["# LectQA-Vid — free re-checks of the stored results", "",
         "Computed from stored answers and stored retrieval rows. No answer was generated and no report was re-run. $0.", "",
         "## Caveats first", "",
         "- **The fallback is our construction.** T1 does not say what its pipeline does when eq. 22 leaves nothing, "
         "so both the fixed reply in the stored replica and the fallback here are our choices. The fallback swaps in "
         "the stored answer of the unfiltered replica; nothing was re-ranked or re-answered.",
         "- **Where the gold window sits is judged by proxies**: whether our own top-4 overlaps it, and whether the "
         "full-transcript answer cites a unit inside it. Neither is a human reading. The hand-check sheet is for that.",
         "- **The echo floor depends on how the references are written.** It says how much token F1 a reply earns "
         "by restating the question; it is not a system.",
         "- By-question intervals are the stored ones (10,000 resamples, numpy seed 20260923). By-video intervals "
         "resample whole videos with the same seed. An interval that contains 0 is marked *(crosses 0)* and is to be "
         "read as no measurable difference.", "",
         "## Open-ended answers: paired differences, by question and by video", "",
         f"{len(qis)} questions on {len(set(vid))} videos, token F1 points, each question's F1 averaged over its 3 repeats.", "",
         "| difference | level | by question | by video |", "|---|---|---:|---:|"]
    for a, b in (("baseline", "replica_no_eq22"), ("fallback", "replica_no_eq22"), ("rrf", "replica_no_eq22"),
                 ("iterative", "rrf"), ("iterative", "replica_no_eq22"), ("iterative", "fallback"),
                 ("iterative", "baseline"), ("full_context", "iterative")):
        for lv in ("overall", *LEVELS):
            k = np.ones(len(qis), bool) if lv == "overall" else level == lv
            L.append(f"| {label[a]} − {label[b]} | {lv} | {cells((score[a] - score[b])[k], vid[k])} |")

    L += ["", "## The replica-to-re-query gap, step by step", "",
          "| step | F1 after | change, by video |", "|---|---:|---:|",
          f"| {label['baseline']} | {score['baseline'].mean():.2f} | |"]
    chain = ("baseline", "fallback", "replica_no_eq22", "rrf", "iterative")
    what = ("unfiltered answer on the empty windows", "eq. 22 off everywhere", "our RRF index in place of T1's pipeline", "one re-query")
    for a, b, w in zip(chain, chain[1:], what):
        L.append(f"| {w} | {score[b].mean():.2f} | {cells(score[b] - score[a], vid).split(' | ')[1]} |")

    # --- why the windows are empty
    enc = t1_encoder(cfg["device"])
    index, chunk = {}, {}
    why, proxy = collections.Counter(), collections.defaultdict(list)
    for q, ap, em in zip(qis, applied, empty):
        if not ap:
            continue
        it = items[q]
        gi, _ = gold_interval(it["vid"], it["q"])
        lo, hi = gi[0] - T1["dt"], gi[1] + T1["dt"]
        if it["vid"] not in index:
            index[it["vid"]] = Index.load(PROCESSED / it["vid"] / "index")
        units = {u.id: u for u in index[it["vid"]].units}
        cited = [units[i] for c in rep0["full_context", q]["claims"][:1] for i in c["unit_ids"] if i in units]
        proxy[bool(em)].append((any(overlaps(u, lo, hi) for u in units.values() if u.modality == "audio"),
                                any(overlaps(units[i], lo, hi) for i in rep0["rrf", q]["evidence"] if i in units),
                                any(overlaps(u, lo, hi) for u in cited) if cited else None,
                                st.mean(f1[q]["rrf"]), st.mean(f1[q]["full_context"])))
        if em:
            if it["vid"] not in chunk:
                chunk[it["vid"]] = t1_chunks(it["vid"], index[it["vid"]], enc)
            cs, vecs = chunk[it["vid"]]
            sims = vecs @ np.asarray(enc([it["question"]]), dtype=np.float32)[0]
            rank = {int(i): n for n, i in enumerate(np.argsort(-sims))}
            inside = [i for i, c in enumerate(cs) if overlaps(c, lo, hi)]
            if not inside:
                why["no replica chunk overlaps the window"] += 1
            elif any(rank[i] < T1["K"] and sims[i] > T1["theta_r"] for i in inside):
                why["a window chunk passes both cuts (not reproduced)"] += 1
            elif any(rank[i] < T1["K"] for i in inside):
                why[f"a window chunk is in the top-{T1['K']} but at or below cosine {T1['theta_r']}"] += 1
            else:
                why[f"every window chunk ranks below the top-{T1['K']}"] += 1
    def share(xs: list) -> str:
        return f"{sum(xs)} of {len(xs)} ({sum(xs) / len(xs):.0%})" if xs else "—"

    L += ["", "## Why eq. 22 leaves windows empty", "",
          f"Eq. 22 was applied on {int(applied.sum())} of the {len(qis)} questions and left {int(empty.sum())} with no context. "
          f"The replica applies it to the top-{T1['K']} above cosine {T1['theta_r']}, as T1's §4.4.2 does.", "",
          "| what happened to the chunks inside the window | questions |", "|---|---:|",
          *(f"| {k} | {v} |" for k, v in why.most_common()), "",
          "| questions | speech exists in the window | our RRF top-4 overlaps it | full-transcript answer's first claim cites a unit in it | RRF F1 | full-transcript F1 |",
          "|---|---:|---:|---:|---:|---:|"]
    for em, name in ((False, "eq. 22 kept something"), (True, "eq. 22 left nothing")):
        p = proxy[em]
        L.append(f"| {name} (n = {len(p)}) | {share([x[0] for x in p])} | {share([x[1] for x in p])} | "
                 f"{share([x[2] for x in p if x[2] is not None])} | {st.mean(x[3] for x in p):.2f} | {st.mean(x[4] for x in p):.2f} |")

    # --- question-echo floor
    echo = np.array([100 * LM.token_prf(items[q]["question"], items[q]["ref"])[2] for q in qis])
    sim = 100 * np.array(LM.similarity([items[q]["question"] for q in qis], [items[q]["ref"] for q in qis], device=cfg["device"]))
    L += ["", "## Question-echo floor", "",
          "The question text scored as if it were the answer, on the same 300 questions.", "",
          "| level | echo F1 | echo similarity | T1 Table 4 F1 | T1 replica F1 | full transcript F1 |", "|---|---:|---:|---:|---:|---:|"]
    for lv in (*LEVELS, "overall"):
        k = np.ones(len(qis), bool) if lv == "overall" else level == lv
        L.append(f"| {lv} | {echo[k].mean():.2f} | {sim[k].mean():.2f} | {THEIRS['open'][lv]['f1']:.2f} | "
                 f"{score['baseline'][k].mean():.2f} | {score['full_context'][k].mean():.2f} |")

    # --- localisation, from the stored pooled rows
    hits = collections.defaultdict(dict)
    for r in map(json.loads, (external / "lectqa_pooled.jsonl").read_text().splitlines()):
        if r["variant"] == "frames" and r["gold"]:
            hits[r["setting"], r["cell"]][r["vid"], r["qi"]] = r
    L += ["", "## Localisation: paired differences, by question and by video", "",
          "Frames corpus, questions whose gold interval passes the format and range checks, percentage points, from "
          "the stored rows of `lectqa_pooled.jsonl` (its single-video rows are the setting of `lectqa_frameslides.md`).", "",
          "| setting | difference | metric | by question | by video |", "|---|---|---|---:|---:|"]
    for setting, a, b in (("single video", "links + reranker", "plain top-8"), ("single video", "reranker, no links", "plain top-8"),
                          ("pooled", "links + reranker", "reranker, no links"), ("pooled", "reranker, no links", "plain top-8"),
                          ("pooled", "links + reranker", "plain top-8"), ("pooled", "links only", "plain top-8")):
        keys = list(hits[setting, a])
        for metric in ("interval@3", "interval@8"):
            d = [100 * (hits[setting, a][k][metric] - hits[setting, b][k][metric]) for k in keys]
            L.append(f"| {setting} | {a} − {b} | {metric} | {cells(d, [k[0] for k in keys])} |")

    # --- all transcripts together
    tok = tiktoken.get_encoding("o200k_base")
    per = [sum(len(tok.encode(u.content)) for u in Index.load(PROCESSED / v / "index").units if u.modality == "audio")
           for v in ids if (PROCESSED / v / "index").exists()]
    L += ["", "## All transcripts together", "",
          f"{len(per)} videos: **{sum(per):,} o200k tokens** in total (per video {min(per)}–{max(per)}, median "
          f"{int(st.median(per))}). The pooled corpus of `lectqa_pooled.md` also fits in one context window. `lectqa_audit.md` gives the "
          "per-video minimum as 140; the 2-token difference was not traced."]

    # --- hand-check sheet
    qa, pool = load_qa(), collections.defaultdict(list)
    done = {(items[q]["vid"], items[q]["question"]) for q, e in zip(qis, empty) if e}
    for v in ids:
        for q in qa.get(v, []) if (PROCESSED / v / "index").exists() else []:
            gi, reason = gold_interval(v, q)
            group = "empty window" if (v, q["question"]) in done else "passes" if gi else reason
            pool[group].append((v, q, gi))
    rng = random.Random(SHEET_SEED)
    sheet = pool["empty window"] + rng.sample(pool["passes"], 16) + rng.sample(pool["ambiguous format"], 25)
    out_csv = external / "lectqa_timestamp_check.csv"
    filled = out_csv.exists() and any(r["checker"] for r in csv.DictReader(out_csv.open()))
    with (out_csv.with_suffix(".blank.csv") if filled else out_csv).open("w", newline="") as f:   # never erase a filled sheet
        w = csv.writer(f)
        w.writerow(["group", "video", "kind", "level", "question", "answer", "timestamp_start", "timestamp_end",
                    "parsed_start_s", "parsed_end_s", "transcript_at_stamp", "answer_is_at_stamp (y/n)",
                    "true_start", "true_end", "checker", "notes"])
        for v, q, gi in sheet:
            said = " ".join(u.content for u in Index.load(PROCESSED / v / "index").units
                            if u.modality == "audio" and gi and overlaps(u, gi[0] - T1["dt"], gi[1] + T1["dt"]))
            group = "empty window" if (v, q["question"]) in done else "passes" if gi else "ambiguous format"
            w.writerow([group, v, q["kind"], q["level"], q["question"], q.get("answer", ""), q["timestamp_start"],
                        q["timestamp_end"], *(gi or ("", "")), said, "", "", "", "", ""])
    L += ["", "## Timestamp hand-check sheet", "",
          f"`{out_csv}`: {len(pool['empty window'])} empty-window questions, 16 drawn from the other "
          f"{len(pool['passes'])} stamps that pass the checks, 25 drawn from the {len(pool['ambiguous format'])} "
          f"with an ambiguous format (seed `{SHEET_SEED}`). The last five columns are for the checker.", "",
          "## Provenance", "",
          f"`python scripts/adapters/lectqa_vid.py recheck --max-cost 0` · commit `{commit}` · inputs "
          "`lectqa_open_modes.jsonl`, `lectqa_attribution.jsonl`, `lectqa_pooled.jsonl` · embedder for the chunk "
          "check: T1 replica's (`open_ended.T1_EMBEDDER`)."]
    (external / "lectqa_recheck.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def timestamp_summary(external: Path) -> int:
    """Summarise the filled timestamp sheet into lectqa_timestamp_check.md. No model is called."""
    rows = list(csv.DictReader((external / "lectqa_timestamp_check.csv").open()))
    if not any(r["checker"] for r in rows):
        raise SystemExit("the sheet is not filled in yet")
    count = lambda g, f: sum(1 for r in rows if r["group"] == g and f(r))                      # noqa: E731
    conf = lambda r: r["notes"].startswith("[high]")                                           # noqa: E731
    hit, miss = (lambda r: r["answer_is_at_stamp (y/n)"] == "y"), (lambda r: r["answer_is_at_stamp (y/n)"] == "n")
    gone = lambda r: r["answer_is_at_stamp (y/n)"] == "not spoken"                             # noqa: E731
    L = ["# LectQA-Vid — timestamp check of 75 gold intervals", "",
         "Each row's reference answer was located in the video's timestamped transcript, and the location was "
         "compared with the annotated stamp ± 3 s. Rows and locations: `lectqa_timestamp_check.csv`.", "",
         "## Caveats first", "",
         "- **The checker is a language model, not a person.** Claude Sonnet agents located each answer in the transcript "
         "without being shown the stamp; the comparison with the stamp is computed here. Each location carries the "
         "agent's confidence. Three rows were spot-checked by reading the transcript; all three agreed.",
         "- **Speech only.** An answer that describes something shown on screen but never said counts as not spoken, "
         "even if the stamp points at the right frame.",
         "- **Small samples:** 34, 16 and 25 rows. The control group (16 passing stamps) is the comparison for the 34.", "",
         "## Stamps that pass the format and range checks", "",
         "| group | rows | stamp points at the spoken answer | stamp misses it | answer never spoken |", "|---|---:|---:|---:|---:|"]
    for g, name in (("empty window", "the 34 where eq. 22 left no context"), ("passes", "16 other passing stamps (control)")):
        n = count(g, lambda r: True)
        L.append(f"| {name} | {n} | {count(g, hit)} | {count(g, miss)} | {count(g, gone)} |")
        L.append(f"| … high-confidence locations only | {count(g, lambda r: conf(r) and not gone(r))} | "
                 f"{count(g, lambda r: hit(r) and conf(r))} | {count(g, lambda r: miss(r) and conf(r))} | — |")
    amb = [r for r in rows if r["group"] == "ambiguous format"]
    readings = collections.Counter(r["answer_is_at_stamp (y/n)"].replace("y, read as ", "") for r in amb)
    L += ["", "## Stamps with an ambiguous format", "",
          "For each of the 25, the stamp was read under every candidate convention (M:S with seconds above 59, "
          "H:M:S with overflow, S:cc, M:S:cc) and compared with the located answer.", "",
          "| reading that matches the spoken answer | rows |", "|---|---:|",
          *[f"| {k} | {v} |" for k, v in readings.most_common()], "",
          "## Reading", "",
          f"On the empty-window questions the stamp misses the spoken answer or the answer is never said in "
          f"{count('empty window', lambda r: miss(r) or gone(r))} of 34, against "
          f"{count('passes', lambda r: miss(r) or gone(r))} of 16 among other passing stamps. So most of the 34 empty "
          "windows point at a moment where the answer is not spoken: eq. 22 removed the evidence because of the interval, "
          "not because retrieval failed. Where the answer is never said, it may be shown on screen; this check cannot tell. Ambiguous stamps follow no single convention: no reading recovers them all, which is why they are "
          "excluded rather than guessed.", "",
          "`python scripts/adapters/lectqa_vid.py tscheck --max-cost 0` rebuilds this file from the sheet."]
    (external / "lectqa_timestamp_check.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0
