"""T1's metric suite on stored answers (`run --metrics theirs`) and the LLM-free benchmark audit."""

from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path


from linkrag.costs import CacheMiss, cached_completer, record_run
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder
from linkrag.link.graph import build_graph
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit, retrieve_linkrag
from linkrag.retrieve.rerank import rerank

from lectqa.common import LEVELS, MCQ_SYSTEM, OPEN_SYSTEM, PROCESSED, RAW, THEIRS, evidence_block, gold_interval, load_qa, mcq_choice, strict_seconds, temporal_links, video_ids
from lectqa.mcq import longest_option


def run(ids: list[str], cfg: dict, dcfg: dict, modes: list[str], out: Path, repeats: int,
        cache_only: bool = False) -> int:
    qa = load_qa()
    llm = cfg["models"]["llm"]
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / str(llm.get("model")),
                                cache_only=cache_only)
    misses = 0
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    k = dcfg["top_k"]
    rcfg, lcfg = cfg["retrieve"]["rerank"], cfg["retrieve"]["linkrag"]
    pool = max(int(rcfg.get("pool", k)), k)
    rows: list[dict] = []
    t_start = time.perf_counter()
    for vid in ids:
        index_dir = PROCESSED / vid / "index"
        if not index_dir.exists():
            print(f"skip {vid}: not prepared")
            continue
        index = Index.load(index_dir)
        units = list(index.units)
        graph = build_graph(units, temporal_links(units))
        for q in qa.get(vid, []):
            for mode in modes:
                for rep in range(repeats):
                    if mode == "baseline":
                        res = [RetrievedUnit(unit=u, score=s, origin="seed")
                               for u, s in retrieve_scored(q["question"], index, encoder=encoder, top_k=k,
                                                           candidates=cfg["retrieve"]["candidates"],
                                                           rrf_k=cfg["retrieve"]["rrf_k"])]
                    else:
                        cand = retrieve_linkrag(q["question"], index, graph, encoder=encoder, mode="linkrag",
                                                k_seed=lcfg["k_seed"], k_final=pool, hops=1,
                                                link_types=["audio_slide"], min_link_score=0.0, decay=lcfg["decay"],
                                                candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"],
                                                normalise_seeds=True, expansion="additive")
                        res = rerank(cand, k, method="complementarity", index=index, graph=graph,
                                     alpha=rcfg["alpha"], beta=rcfg["beta"], gamma=rcfg["gamma"],
                                     question=q["question"], device=cfg["device"])
                    ev = evidence_block(res)
                    if q["kind"] == "mcq":
                        opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(q["options"]))
                        try:
                            reply = complete(MCQ_SYSTEM, f"Evidence:\n{ev}\n\nQuestion: {q['question']}\n{opts}\n\nLetter:")
                        except CacheMiss:
                            misses += 1
                            continue
                        choice = mcq_choice(reply, q["options"])
                        gold = q["options"].index(q["answer"]) if q["answer"] in q["options"] else None
                        rows.append({"vid": vid, "kind": "mcq", "level": q["level"], "mode": mode, "rep": rep,
                                     "correct": float(choice is not None and choice == gold), "reply": reply[:40],
                                     "pred_label": None if choice is None else "ABCD"[choice],
                                     "gold_label": None if gold is None else "ABCD"[gold]})
                    else:
                        try:
                            reply = complete(OPEN_SYSTEM, f"Evidence:\n{ev}\n\nQuestion: {q['question']}\n\nAnswer:")
                        except CacheMiss:
                            misses += 1
                            continue
                        rows.append({"vid": vid, "kind": "open", "level": q["level"], "mode": mode, "rep": rep,
                                     "reply": reply, "ref": q["answer"]})
        done = len({r["vid"] for r in rows})
        print(f"{vid}: {len(qa.get(vid, []))} QA · {done} videos done · {time.perf_counter() - t_start:.0f}s")

    if not rows:
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    footer = record_run("scripts/adapters/lectqa_vid.py", f"lectqa_vid their metrics ({len({r['vid'] for r in rows})} videos)",
                        [(str(llm.get("model")), complete.usage)], cfg["models"].get("pricing"))
    return theirs_report(rows, ids, cfg, llm, modes, out, misses, footer)


def theirs_report(rows: list[dict], ids: list[str], cfg: dict, llm: dict, modes: list[str], out: Path,
                  misses: int, footer: list[str]) -> int:
    """Their metric suite (eqs. 29-36, linkrag.eval.lectqa_metrics), per difficulty and overall,
    next to their published Table 4 / Table 5 rows."""
    from linkrag.eval import lectqa_metrics as LM
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_videos = len({r["vid"] for r in rows})
    L = ["# LectQA-Vid — their metric suite (T1 eqs. 29–36)", "",
         f"{n_videos} videos · answerer `{llm.get('model')}` (cache-only replay of stored answers; "
         f"{misses} cache misses skipped) · metrics `linkrag.eval.lectqa_metrics`: set-based token P/R/F1, "
         f"nltk BLEU-4 (no smoothing) and METEOR (alpha 0.9, beta 3, gamma 0.5), ROUGE-1 = unigram recall, "
         f"similarity = all-MiniLM-L6-v2 cosine, all in %.", "",
         "**Not a like-for-like comparison.** Their answering LLM is unnamed (§4.4.4). Ours is far "
         "stronger, and their split and 1,000-pair evaluation subset are unpublished, so these are all "
         "QA pairs of the videos processed. The baseline-vs-linkrag rows are the retrieval comparison.", "",
         "## Open-ended (their Table 4)", "",
         "| level | mode | n | F1 | Sim | BLEU | METEOR | R1 | token P | token R |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in modes:
            sel = [r for r in rows if r["kind"] == "open" and r["mode"] == mode and (level == "overall" or r["level"] == level)]
            if not sel:
                continue
            s = LM.score_open([r["reply"] for r in sel], [r["ref"] for r in sel], device=cfg["device"])
            L.append(f"| {level} | {mode} | {s['n']} | {100*s['f1']:.2f} | {100*s['sim']:.2f} | {100*s['bleu']:.2f} | "
                     f"{100*s['meteor']:.2f} | {100*s['rouge1']:.2f} | {100*s['precision']:.2f} | {100*s['recall']:.2f} |")
        th = THEIRS["open"][level]
        L.append(f"| {level} | **theirs (Table 4)** | — | {th['f1']:.2f} | {th['sim']:.2f} | {th['bleu']:.2f} | "
                 f"{th['meteor']:.2f} | {th['rouge1']:.2f} | — | — |")
    L += ["", "## MCQ (their Table 5)", "", "| level | mode | n | ACC | Precision | Recall | F1 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in modes:
            sel = [r for r in rows if r["kind"] == "mcq" and r["mode"] == mode and (level == "overall" or r["level"] == level)]
            if not sel:
                continue
            s = LM.score_mcq([r["pred_label"] for r in sel], [r["gold_label"] for r in sel])
            L.append(f"| {level} | {mode} | {s['n']} | {100*s['accuracy']:.2f} | {100*s['precision']:.2f} | "
                     f"{100*s['recall']:.2f} | {100*s['f1']:.2f} |")
        L.append(f"| {level} | **theirs (Table 5)** | — | {THEIRS['mcq'][level]:.2f} | — | — | — |")
    gold = collections.Counter(r["gold_label"] for r in rows if r["kind"] == "mcq")
    top, top_n = gold.most_common(1)[0] if gold else (None, 0)
    if gold and top_n / sum(gold.values()) > 0.9:
        L += ["", f"**MCQ caveat: not reportable in this form.** The gold option is `{top}` for {top_n} of "
              f"{sum(gold.values())} questions. The published file lists the correct answer first for 1,484 of "
              "1,489 MCQs, and options were presented in that stored order, so accuracy rewards any preference "
              "for the first option. Macro P/R/F1 are degenerate with one gold class. Their Table 5 P/R/F1 ≈ ACC "
              "suggests shuffled options there; the paper does not say. The valid measurement is a re-run with "
              "options shuffled under a fixed seed, which needs new LLM calls."]
    n_by = collections.Counter((r["kind"], r["mode"]) for r in rows)
    L += ["", "Counts: " + ", ".join(f"{k} {m} {n}" for (k, m), n in sorted(n_by.items()))
          + f"; cache misses skipped: {misses}."]
    L += footer
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def audit(out: Path) -> int:
    """LLM-free audit of the published benchmark and of what we could obtain from it."""
    import av
    import statistics as st
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")          # the GPT-4.1 / GPT-5 family encoding
    mcq_raw = json.loads((RAW / "mcq_questions.json").read_text())
    qa = load_qa()
    status = json.loads((RAW / "fetch_status.json").read_text())
    obtained = [v for v in video_ids(None, None) if status.get(v, {}).get("status") == "obtained"]
    L = ["# LectQA-Vid — benchmark audit", "",
         "LLM-free. Published files: `mcq_questions.json`, `open_ended_questions.json`, `video_links.docx` "
         "(Mendeley doi:10.17632/yt4nmz9mcv.1). Obtained media: YouTube, 2026-09-23.", ""]

    # (a) MCQ answer position
    pos, by_level = collections.Counter(), collections.defaultdict(collections.Counter)
    longest = collections.defaultdict(collections.Counter)
    for vid, qs in mcq_raw.items():
        for q in qs:
            letter = "ABCD"[q["options"].index(q["answer"])] if q["answer"] in q["options"] else "not an option"
            pos[letter] += 1
            by_level[q["level"].replace("_", " ")][letter] += 1
            longest[q["level"].replace("_", " ")][longest_option(q["options"]) == letter] += 1
    n_mcq = sum(pos.values())
    L += ["## (a) MCQ answer position", "",
          f"All {n_mcq} published MCQs. The correct answer's position in the stored option list:", "",
          "| position | count | share |", "|---|---:|---:|",
          *[f"| {k} | {v} | {v / n_mcq:.1%} |" for k, v in sorted(pos.items())], "",
          f"**A constant-\"A\" answerer scores {pos['A'] / n_mcq:.1%}** "
          + "(" + ", ".join(f"{lv} {c['A'] / sum(c.values()):.1%}" for lv, c in sorted(by_level.items())) + "), "
          "against their reported 53.43 %. Any MCQ run that keeps the stored order measures position "
          "preference, not retrieval. Options must be shuffled.", "",
          "**Shuffling is not enough: the answer is the longest option.** The gold is the unique longest of the "
          "four options in:", "", "| level | MCQs | gold = unique longest option |", "|---|---:|---:|",
          *[f"| {lv} | {sum(c.values())} | {c[True]} ({c[True] / sum(c.values()):.1%}) |"
            for lv, c in sorted(longest.items())],
          f"| **all** | {n_mcq} | {sum(c[True] for c in longest.values())} "
          f"({sum(c[True] for c in longest.values()) / n_mcq:.1%}) |", "",
          "So an answerer can score near these rates from the options alone, with no video, whatever the order.", ""]

    # (b) timestamps
    shapes, strict = collections.Counter(), collections.Counter()
    for vid, qs in qa.items():
        for q in qs:
            for k in ("timestamp_start", "timestamp_end"):
                t = str(q[k]).strip()
                shapes[re.sub(r"\d", "9", t)] += 1
                strict["unambiguous" if strict_seconds(t) is not None else "ambiguous"] += 1
    reasons, by_range = collections.Counter(), collections.defaultdict(collections.Counter)
    for vid in obtained:
        n = int(vid.split("_")[1])
        rng = "1–35" if n <= 35 else "36–63" if n <= 63 else "64–100"
        for q in qa.get(vid, []):
            why = gold_interval(vid, q)[1]
            reasons[why] += 1
            by_range[rng][why] += 1
    n_q = sum(reasons.values())
    L += ["## (b) Gold timestamps", "",
          f"Every start/end stamp of all {sum(len(v) for v in qa.values())} published QA pairs, by digit shape "
          "(9 = any digit):", "", "| shape | stamps |", "|---|---:|",
          *[f"| `{k}` | {v} |" for k, v in shapes.most_common()], "",
          f"Unambiguous by `strict_seconds` (HH:MM:SS / MM:SS with fields < 60, or plain seconds): "
          f"{strict['unambiguous']} of {sum(strict.values())} stamps. `00:12:70` (= 12.70 s), `258:36` and "
          "`01:91:60` also occur, and their meaning differs between videos.", "",
          f"Gold intervals of the {n_q} QA pairs of the {len(obtained)} obtained videos: "
          + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ".", "",
          "| videos | scorable | ambiguous | past the video's end | other |", "|---|---:|---:|---:|---:|"]
    for rng in ("1–35", "36–63", "64–100"):
        c = by_range[rng]
        other = sum(v for k, v in c.items() if k not in ("ok", "ambiguous format", "beyond the fetched video's end"))
        past = c["beyond the fetched video's end"]
        L.append(f"| {rng} | {c['ok']} | {c['ambiguous format']} | {past} | {other} |")
    L.append("")

    # (c) links
    dead = {v: s for v, s in status.items() if s.get("status") == "dead"}
    L += ["## (c) Link availability", "",
          f"{len(obtained)}/100 obtained, {len(dead)} dead. Mendeley ships no media or transcripts, so every video "
          "comes from its YouTube link.", "", "| video | reason |", "|---|---|",
          *[f"| {v} | {s.get('error', '')[:90]} |" for v, s in sorted(dead.items(), key=lambda kv: int(kv[0].split('_')[1]))], ""]

    # (d) genre
    rows, cls = [], collections.Counter()
    for vid in obtained:
        f = PROCESSED / vid / "index_frameslides" / "frameslides_stats.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        with av.open(str(RAW / "videos" / f"{vid}.m4a")) as c:
            dur = float(c.duration) / 1e6
        rate = s["slides"] / (dur / 60)
        kind = "slide talk" if rate <= 3 else "animated explainer" if rate > 6 else "mixed"
        gate = (s.get("gate") or {}).get("related")
        cls[(kind, "accepted" if gate else "rejected")] += 1
        rows.append(f"| {vid} | {dur:.0f} | {s['slides']} | {rate:.1f} | {'accepted' if gate else 'rejected'} | {kind} |")
    L += ["## (d) Genre", "",
          "**Rule, stated before looking at answers:** slide changes per minute from the frame-derived deck "
          "(1 fps dHash segmentation). **≤ 3 per minute = slide talk** (a slide stays up ≥ 20 s), "
          "**> 6 = animated explainer**, otherwise mixed. The relatedness-gate verdict (does the speech align "
          "with the recovered deck) is reported next to it, not folded into the rule.", "",
          "| class | gate accepted | gate rejected |", "|---|---:|---:|",
          *[f"| {k} | {cls[(k, 'accepted')]} | {cls[(k, 'rejected')]} |" for k in ("slide talk", "mixed", "animated explainer")],
          f"| **total** | {sum(v for (k, g), v in cls.items() if g == 'accepted')} | "
          f"{sum(v for (k, g), v in cls.items() if g == 'rejected')} |", "",
          "<details><summary>per video</summary>", "", "| video | duration s | slides | slides/min | gate | class |",
          "|---|---:|---:|---:|---|---|", *rows, "", "</details>", ""]

    # (e) context fit
    toks = {}
    for vid in obtained:
        units = [u for u in Index.load(PROCESSED / vid / "index").units if u.modality == "audio"]
        text = " ".join(u.content for u in sorted(units, key=lambda u: u.location.start_s or 0.0))
        toks[vid] = len(enc.encode(text))
    vals = sorted(toks.values())
    L += ["## (e) Context fit", "",
          f"Full transcript per obtained video, o200k tokens (GPT-4.1 / GPT-5 encoding), n = {len(vals)}: "
          f"min {vals[0]}, median {st.median(vals):.0f}, max {vals[-1]}.", "",
          "| window | videos whose whole transcript fits |", "|---|---:|",
          *[f"| {w:,} | {sum(v <= w for v in vals)}/{len(vals)} |" for w in (2048, 4096, 8192, 32768)], "",
          "The API does not report the answerer's (gpt-5.4-mini-2026-03-17) context window. The largest "
          "transcript is far below any current OpenAI window, so a no-retrieval full-transcript answerer is a "
          "valid baseline on this benchmark.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0
