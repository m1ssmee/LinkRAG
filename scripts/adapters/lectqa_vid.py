#!/usr/bin/env python3
"""LectQA-Vid adapter -- priority (iii). Target T1 (Shafiq et al., CMC 88(2), 2026).

The published dataset (Mendeley, doi:10.17632/yt4nmz9mcv.1, CC BY 4.0) ships the QA
pairs and YouTube links only -- "videos, keyframes and transcripts are not provided
because of copyright issues" -- so this adapter rebuilds the inputs their pipeline
consumed: audio -> Whisper transcript; keyframes -> OCR (they used Gemini captions;
we use tesseract, optionally the VLM captioner). Their 80/10/10 split and 1,000-pair
evaluation subset are not published; results are reported on every QA pair of the
videos processed, with n stated.

    python scripts/adapters/lectqa_vid.py fetch   --videos 10        # yt-dlp: audio m4a + video-only mp4
    python scripts/adapters/lectqa_vid.py prepare --videos 10        # whisper + frames + OCR -> per-video index
    python scripts/adapters/lectqa_vid.py run     --videos 10 --modes baseline,linkrag

Metrics (their Tables 4/5): open-ended token-F1 and ROUGE-1 (SQuAD-style
normalisation) and a semantic similarity (cosine of bge-m3 embeddings -- they do not
name their embedder, so this column is not strictly comparable); MCQ accuracy by
option match. Per difficulty level and Overall.

What "linkrag" means on a single video: the deck IS the video, so the alignment link
is temporal co-occurrence (audio segment <-> frames on screen at that time) -- the
same signal T1 uses -- plus additive expansion and the complementarity reranker over
the transcript + frame-OCR units. There is no cross-file link to make here; the
cross-file setting is the extended dataset (priority vi).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import string
import sys
import time
from pathlib import Path

import numpy as np

from linkrag.core import EvidenceUnit, Link, Location, load_config, refuse_strong_in_batch, set_max_cost, setup_logging
from linkrag.costs import CacheMiss, cached_completer, price_for, record_run, usage_cost
from linkrag.generate.answer import http_completer
from linkrag.index import Index, build_index, default_encoder
from linkrag.link.align import load_links, save_links
from linkrag.link.graph import build_graph
from linkrag.manifest import MANIFEST_NAME, build_manifest, write_manifest
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit, retrieve_linkrag
from linkrag.retrieve.rerank import rerank

RAW = Path("data/raw/lectqa_vid")
PROCESSED = Path("data/processed/lectqa_vid")
LEVELS = ("simple", "hard", "very hard")
# Their published numbers, all in %: CMC 88(2) art. 96, full-text HTML, Tables 4-6 (images),
# read 2026-09-23. Corrected then: earlier values here (sim 0.71, MCQ 56.30 %, and most
# per-difficulty cells) did not match the published tables.
THEIRS = {
    "open": {  # Table 4, "Temporal-Aware RAG (Ours)"
        "simple":    {"f1": 29.47, "sim": 77.23, "bleu": 8.61, "meteor": 38.15, "rouge1": 36.82},
        "hard":      {"f1": 24.38, "sim": 74.56, "bleu": 5.29, "meteor": 34.71, "rouge1": 30.94},
        "very hard": {"f1": 16.72, "sim": 71.48, "bleu": 2.83, "meteor": 24.36, "rouge1": 22.51},
        "overall":   {"f1": 23.52, "sim": 74.42, "bleu": 5.58, "meteor": 32.41, "rouge1": 29.76},
    },
    "mcq": {"simple": 57.29, "hard": 52.34, "very hard": 50.67, "overall": 53.43},  # Table 5, ACC
    # Table 6, "Multimodal RAG (ASR + captions, no timestamps)" -- their no-temporal baseline
    "open_no_timestamps": {"overall": {"f1": 14.80, "sim": 61.00, "bleu": 2.95, "meteor": 21.65}},
}


# ----------------------------------------------------------------- data

def video_links() -> dict[str, str]:
    import docx
    doc = docx.Document(RAW / "video_links.docx")
    out = {}
    for row in doc.tables[0].rows[1:]:
        vid, url = (c.text.strip() for c in row.cells[:2])
        out[vid.replace(" ", "_").rstrip(".")] = url      # the docx labels one row "video_94."
    return out


def load_qa() -> dict[str, list[dict]]:
    mcq = json.loads((RAW / "mcq_questions.json").read_text())
    opn = json.loads((RAW / "open_ended_questions.json").read_text())
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for vid, qs in mcq.items():
        for q in qs:
            out[vid].append({**q, "kind": "mcq", "level": q["level"].replace("_", " ")})
    for vid, qs in opn.items():
        for q in qs:
            out[vid].append({**q, "kind": "open", "level": q["level"].replace("_", " ")})
    return out


def video_ids(n: int | None, only: str | None) -> list[str]:
    ids = [f"video_{i}" for i in range(1, 101)]
    if only:
        ids = [v for v in ids if v in set(only.split(","))]
    return ids[:n] if n else ids


# ----------------------------------------------------------------- fetch

def fetch(ids: list[str]) -> None:
    """Two files per video, no ffmpeg merge: audio (m4a, whisper reads it directly)
    and a video-only mp4 (frames). YouTube no longer serves progressive streams."""
    import yt_dlp
    links = video_links()
    (RAW / "videos").mkdir(parents=True, exist_ok=True)
    status_path = RAW / "fetch_status.json"          # per video, merged across runs
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    failures = []
    for vid in ids:
        audio, video = RAW / "videos" / f"{vid}.m4a", RAW / "videos" / f"{vid}.video.mp4"
        wanted = [(audio, "140/bestaudio[ext=m4a]/bestaudio"),
                  (video, "134/160/bestvideo[ext=mp4][height<=360]/bestvideo[height<=360]")]
        for out, fmt in wanted:
            if out.exists():
                continue
            opts = {"format": fmt, "outtmpl": str(out), "quiet": True, "noprogress": True, "no_warnings": True}
            try:
                with yt_dlp.YoutubeDL(opts) as y:
                    y.download([links[vid]])
            except Exception as exc:  # a private/removed video must not stop the batch
                failures.append((vid, str(exc).splitlines()[-1][:120]))
                status[vid] = {"status": "dead", "source": "youtube", "url": links[vid], "error": failures[-1][1]}
                break
        else:
            status[vid] = {"status": "obtained", "source": "youtube", "url": links[vid]}
            print(f"fetched {vid}")
        status_path.write_text(json.dumps(dict(sorted(status.items(), key=lambda kv: int(kv[0].split("_")[1]))), indent=1))
    if failures:
        (RAW / "fetch_failures.txt").write_text("\n".join(f"{v}\t{e}" for v, e in failures) + "\n")
        print(f"{len(failures)} video(s) unavailable -> {RAW / 'fetch_failures.txt'}", file=sys.stderr)


# ----------------------------------------------------------------- prepare

def extract_frames(video: Path, out_dir: Path, every_s: float, min_change: float = 0.08) -> list[tuple[float, Path]]:
    """One frame every `every_s` seconds, kept only if it differs from the last kept
    frame (mean absolute pixel difference on a 64x36 grey thumbnail > min_change)."""
    import av
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[float, Path]] = []
    last_thumb = None
    next_t = 0.0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if t + 1e-6 < next_t:
                continue
            next_t = t + every_s
            img = frame.to_image()
            thumb = np.asarray(img.convert("L").resize((64, 36)), dtype=np.float32) / 255.0
            if last_thumb is not None and float(np.abs(thumb - last_thumb).mean()) < min_change:
                continue
            last_thumb = thumb
            path = out_dir / f"{video.stem}_{int(round(t)):05d}.png"
            if not path.exists():
                img.save(path)
            kept.append((t, path))
    return kept


def prepare(ids: list[str], cfg: dict, dcfg: dict) -> None:
    from linkrag.ingest.audio import ingest_audio
    from linkrag.ingest.image import ocr
    encoder = None
    for vid in ids:
        audio_path, video = RAW / "videos" / f"{vid}.m4a", RAW / "videos" / f"{vid}.video.mp4"
        index_dir = PROCESSED / vid / "index"
        if not audio_path.exists() or not video.exists():
            print(f"skip {vid}: not fetched")
            continue
        if index_dir.exists():
            continue
        t0 = time.perf_counter()
        frozen = PROCESSED / vid / "transcript.frozen.json"     # whisper runs once per video
        audio = ingest_audio(audio_path, model_size=cfg["models"]["whisper"], device=cfg["device"],
                             compute_type=cfg["models"]["whisper_compute_type"],
                             window_seconds=dcfg["audio_segment_seconds"], segmentation="sentence",
                             transcript=frozen if frozen.exists() else None,
                             freeze_to=None if frozen.exists() else frozen)
        frames = extract_frames(video, PROCESSED / vid / "frames", dcfg["frame_interval_s"])
        units = list(audio)
        for i, (t, path) in enumerate(frames):
            end = frames[i + 1][0] if i + 1 < len(frames) else t + dcfg["frame_interval_s"]
            text = " ".join(ocr(path).split())
            units.append(EvidenceUnit(id=f"{vid}:f{i}", modality="figure", content=text,
                                      source_file=str(audio_path), location=Location(start_s=t, end_s=end),
                                      metadata={"frame_path": str(path), "content_source": "ocr",
                                                "slide_deck": False}))
        if encoder is None:
            encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
            encoder([""])
        index = build_index(units, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                            device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"])
        index.save(index_dir)
        write_manifest(build_manifest([audio_path, video], units), PROCESSED / vid / MANIFEST_NAME)
        print(f"prepared {vid}: {len(audio)} audio + {len(frames)} frames ({time.perf_counter() - t0:.0f}s)")


# ----------------------------------------------------------------- linking (temporal)

def temporal_links(units: list[EvidenceUnit]) -> list[Link]:
    """audio segment <-> every frame on screen during it. This is T1's own signal;
    on a single video it is the only alignment there is."""
    audio = [u for u in units if u.modality == "audio"]
    frames = [u for u in units if u.modality == "figure"]
    links = []
    for a in audio:
        for f in frames:
            if f.location.start_s < a.location.end_s and f.location.end_s > a.location.start_s:
                overlap = min(a.location.end_s, f.location.end_s) - max(a.location.start_s, f.location.start_s)
                links.append(Link(a.id, f.id, "audio_slide", round(min(1.0, overlap / max(1.0, f.location.end_s - f.location.start_s)), 4)))
    return links


# ----------------------------------------------------------------- answering + scoring

MCQ_SYSTEM = ("You answer a multiple-choice question about a lecture video using ONLY the evidence. "
              "Reply with the single letter of the best option and nothing else.")
OPEN_SYSTEM = ("You answer a question about a lecture video using ONLY the evidence. Answer in one or "
               "two sentences. If the evidence does not contain the answer, say: NOT ANSWERABLE.")


def evidence_block(results: list[RetrievedUnit]) -> str:
    lines = []
    for r in results:
        u = r.unit
        tag = f"[{u.modality.upper()} {int(u.location.start_s // 60)}:{int(u.location.start_s % 60):02d}]"
        lines.append(f"{tag} {' '.join(u.content.split()) or '(no text on frame)'}")
    return "\n".join(lines)


def _norm(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(pred: str, ref: str) -> float:
    p, r = _norm(pred), _norm(ref)
    if not p or not r:
        return float(p == r)
    common = collections.Counter(p) & collections.Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge1_raw(pred: str, ref: str) -> float:
    p = re.findall(r"\w+", pred.lower())
    r = re.findall(r"\w+", ref.lower())
    if not p or not r:
        return 0.0
    n = sum((collections.Counter(p) & collections.Counter(r)).values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def mcq_choice(reply: str, options: list[str]) -> int | None:
    m = re.match(r"\s*\(?([A-Da-d])\)?[\s.:)]*", reply)
    if m:
        return "ABCD".index(m.group(1).upper())
    low = reply.strip().lower()
    for i, o in enumerate(options):
        if low == o.strip().lower():
            return i
    return None


def run(ids: list[str], cfg: dict, dcfg: dict, modes: list[str], out: Path, repeats: int,
        metrics: str = "ours", cache_only: bool = False) -> int:
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
                        vec = np.asarray(encoder([reply, q["answer"]]), dtype=np.float32)
                        sim = float(vec[0] @ vec[1] / (np.linalg.norm(vec[0]) * np.linalg.norm(vec[1]) + 1e-9))
                        rows.append({"vid": vid, "kind": "open", "level": q["level"], "mode": mode, "rep": rep,
                                     "f1": token_f1(reply, q["answer"]), "rouge1": rouge1_raw(reply, q["answer"]),
                                     "sim": sim, "reply": reply if metrics == "theirs" else reply[:200],
                                     "ref": q["answer"]})
        done = len({r["vid"] for r in rows})
        print(f"{vid}: {len(qa.get(vid, []))} QA · {done} videos done · {time.perf_counter() - t_start:.0f}s")

    if not rows:
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    if metrics == "theirs":
        footer = record_run("scripts/adapters/lectqa_vid.py", f"lectqa_vid their metrics ({len({r['vid'] for r in rows})} videos)",
                            [(str(llm.get("model")), complete.usage)], cfg["models"].get("pricing"))
        return theirs_report(rows, ids, cfg, llm, modes, out, misses, footer)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def agg(kind, mode, level, key):
        xs = [r[key] for r in rows if r["kind"] == kind and r["mode"] == mode and (level == "overall" or r["level"] == level)]
        return (statistics.mean(xs), len(xs)) if xs else (float("nan"), 0)

    n_videos = len({r["vid"] for r in rows})
    L = ["# LectQA-Vid — target T1 — first run", "",
         f"{n_videos} of 100 videos · {len(rows) // (len(modes) * repeats)} QA pairs per mode · "
         f"modes {modes} · k={k} · answerer `{llm.get('model')}` temperature={llm.get('temperature')} · "
         f"repeats {repeats} · transcript: whisper-{cfg['models']['whisper']} · frames every "
         f"{dcfg['frame_interval_s']}s, OCR tesseract (their pipeline used Whisper large-v3 and Gemini captions)", "",
         "Their split and 1,000-pair evaluation subset are unpublished; this is every QA pair of the "
         "videos processed. Semantic similarity here is bge-m3 cosine; they do not name their embedder.", "",
         "**Read with care.** The answerer here (`" + str(llm.get("model")) + "`) is far stronger than the "
         "open models T1 evaluated with, so a gap to their numbers is mostly the LLM, not retrieval; the "
         "baseline-vs-linkrag rows are the retrieval comparison. Videos are 2–5 minutes, so top-4 of "
         "~15–40 units already covers much of each video. MCQ distractors are weak (see accuracy).", "",
         "## Open-ended (their Table 4 / 6)", "",
         "| level | n | mode | token-F1 | ROUGE-1 | sim (bge-m3) | their F1 | their ROUGE-1 | their sim |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in modes:
            f1, n = agg("open", mode, level, "f1")
            if not n:
                continue
            r1, _ = agg("open", mode, level, "rouge1")
            sm, _ = agg("open", mode, level, "sim")
            th = THEIRS["open"][level]
            tf, tr, ts = th["f1"], th["rouge1"], th["sim"] / 100
            L.append(f"| {level} | {n} | {mode} | {f1:.1%} | {r1:.1%} | {sm:.2f} | {tf:.2f}% | {tr:.2f}% | {ts:.2f} |")
    L += ["", "## MCQ (their Table 5)", "", "| level | n | mode | accuracy | their accuracy |", "|---|---:|---|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in modes:
            acc, n = agg("mcq", mode, level, "correct")
            if n:
                L.append(f"| {level} | {n} | {mode} | {acc:.1%} | {THEIRS['mcq'][level]:.2f}% |")
    footer = record_run("scripts/adapters/lectqa_vid.py", f"lectqa_vid first run ({n_videos} videos)",
                        [(str(llm.get("model")), complete.usage)], cfg["models"].get("pricing"))
    L += footer
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[-12:]))
    print(f"wrote {out}")
    return 0


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


# ----------------------------------------------------------------- v2: localisation

def strict_seconds(t: str) -> float | None:
    """A timestamp only when its format is unambiguous: HH:MM:SS or MM:SS with minute and
    second fields below 60, or plain seconds. The published files also contain SS:cc
    ('00:12:70'), SSS:cc ('258:36') and M:SSS:cc ('01:91:60') stamps whose meaning differs
    between videos, so those are not guessed at."""
    parts = str(t).strip().split(":")
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) > 3 or any(v >= 60 for v in vals[1:]) or (len(vals) == 3 and vals[0] >= 24):
        return None
    while len(vals) < 3:
        vals.insert(0, 0.0)
    return vals[0] * 3600 + vals[1] * 60 + vals[2]


_DURATION: dict[str, float] = {}


def gold_interval(vid: str, q: dict) -> tuple[tuple[float, float] | None, str]:
    """(interval, reason). None when the stamp is ambiguous or outside the fetched video."""
    if vid not in _DURATION:
        import av
        with av.open(str(RAW / "videos" / f"{vid}.m4a")) as c:
            _DURATION[vid] = float(c.duration) / 1e6
    g0, g1 = strict_seconds(q["timestamp_start"]), strict_seconds(q["timestamp_end"])
    if g0 is None or g1 is None:
        return None, "ambiguous format"
    if g1 < g0:
        return None, "end before start"
    if g1 > _DURATION[vid] + 2.0:
        return None, "beyond the fetched video's end"
    return (g0, g1 if g1 > g0 else g0 + 1.0), "ok"


def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def retrieve_pool(mode, q, index, graph, *, encoder, cfg, complete, pool, link_types=("audio_slide",)):
    link_types = list(link_types) if link_types else None
    lcfg, icfg = cfg["retrieve"]["linkrag"], cfg["retrieve"]["iterative"]
    common = dict(candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"])
    if mode == "baseline":
        return [RetrievedUnit(unit=u, score=s, origin="seed")
                for u, s in retrieve_scored(q, index, encoder=encoder, top_k=pool, **common)]
    if mode == "linkrag":
        return retrieve_linkrag(q, index, graph, encoder=encoder, mode="linkrag", k_seed=lcfg["k_seed"], k_final=pool,
                                hops=1, link_types=link_types, min_link_score=0.0, decay=lcfg["decay"],
                                normalise_seeds=True, expansion="additive", **common)
    from linkrag.retrieve.iterative import retrieve_iterative, retrieve_linkrag_iter
    if mode == "iterative":
        return retrieve_iterative(q, index, encoder=encoder, complete=complete, rounds=icfg["rounds"],
                                  k_per_round=icfg["k_per_round"], k_final=pool, **common).units
    out, _ = retrieve_linkrag_iter(q, index, graph, encoder=encoder, complete=complete, rounds=icfg["rounds"],
                                   k_seed=lcfg["k_seed"], k_final=pool, hops=1, link_types=link_types,
                                   min_link_score=0.0, decay=lcfg["decay"], normalise_seeds=True,
                                   expansion="additive", **common)
    return out


def localise(ids: list[str], cfg: dict, dcfg: dict, *, index_name: str = "index", label: str = "",
             cache_only: bool = True, max_cost: float = 1.0,
             modes: tuple[str, ...] = ("baseline", "linkrag", "iterative", "linkrag_iter"),
             link_types: tuple[str, ...] | None = ("audio_slide",)) -> tuple[list[dict], dict]:
    """Temporal localisation: does a retrieved unit overlap the gold interval? No answerer.
    The iterative modes need one LLM call per question for the follow-up query; with
    `cache_only` (default) those cells are filled only where that call was already
    paid for, and are marked `not run` otherwise -- no new spend."""
    from linkrag.retrieve.rerank import rerank
    qa = load_qa()
    llm = cfg["models"]["llm"]
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / str(llm.get("model")),
                                cache_only=cache_only, max_cost_usd=max_cost,
                                price=price_for(str(llm.get("model")), cfg["models"].get("pricing")))
    skipped = 0
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    rcfg = cfg["retrieve"]["rerank"]
    ks, pool = (1, 3, 8), 20
    rows = []
    excluded: collections.Counter = collections.Counter()
    for vid in ids:
        index_dir = PROCESSED / vid / index_name
        if not index_dir.exists():
            continue
        index = Index.load(index_dir)
        units = list(index.units)
        links_file = index_dir / "links.jsonl"            # an index with its own linking layer
        graph = build_graph(units, load_links(links_file) if links_file.exists() else temporal_links(units))
        for q in qa.get(vid, []):
            gi, why = gold_interval(vid, q)
            excluded[why] += 1
            if gi is None:
                continue
            g0, g1 = gi
            for mode in modes:
                try:
                    cand = retrieve_pool(mode, q["question"], index, graph, encoder=encoder, cfg=cfg, complete=complete,
                                         pool=pool, link_types=link_types)
                except CacheMiss:
                    skipped += 1
                    continue
                for method in ("none", "complementarity", "complementarity+gate"):
                    picked = rerank(cand, 8, method=method.split("+")[0], index=index, graph=graph, alpha=rcfg["alpha"],
                                    beta=rcfg["beta"], gamma=rcfg["gamma"], question=q["question"], device=cfg["device"],
                                    modality_gate=method.endswith("+gate"))
                    ious = [max(iou(a, b, g0, g1) for a, b in
                                (r.unit.metadata.get("intervals") or [(r.unit.location.start_s, r.unit.location.end_s)]))
                            for r in picked]
                    g = rerank.last_gate
                    row = {"vid": vid, "kind": q["kind"], "level": q["level"], "mode": mode, "rerank": method,
                           "best_iou": max(ious) if ious else 0.0,
                           "gate": (("concentrated" if g["concentrated"] else "spread") if g else "off")}
                    for k in ks:
                        row[f"hit@{k}"] = float(any(x > 0 for x in ious[:k]))
                    row["mods"] = dict(collections.Counter(r.unit.modality for r in picked))
                    rows.append(row)
        print(f"{label} {vid}: {len(qa.get(vid, []))} QA localised")
    return rows, {"llm_usage": complete.usage, "model": str(llm.get("model")), "skipped_cells": skipped,
                  "gold": dict(excluded)}


WITH_GATE = ("none", "complementarity", "complementarity+gate")


def localisation_table(rows: list[dict], title: str, methods: tuple[str, ...] = ("none", "complementarity")) -> list[str]:
    """hit@k and mean best IoU per difficulty x mode x rerank method."""
    import statistics as st
    L = [f"### {title}", "",
         "| level | n | mode | rerank | hit@1 | hit@3 | hit@8 | mean best IoU |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in ("baseline", "linkrag", "iterative", "linkrag_iter"):
            for method in methods:
                xs = [r for r in rows if r["mode"] == mode and r["rerank"] == method and (level == "overall" or r["level"] == level)]
                if not xs:
                    continue
                m = lambda k: st.mean(r[k] for r in xs)
                L.append(f"| {level} | {len(xs)} | {mode} | {method} | {m('hit@1'):.1%} | {m('hit@3'):.1%} | {m('hit@8'):.1%} | {m('best_iou'):.3f} |")
    return L


# ----------------------------------------------------------------- MCQ, shuffled options

MCQ_SEED = "lectqa-mcq-shuffle-20260923"


def longest_option(options: list[str]) -> str | None:
    """Letter of the unique longest option (None on a tie)."""
    n = [len(o) for o in options]
    return "ABCD"[n.index(max(n))] if n.count(max(n)) == 1 else None


def shuffled(q: dict, vid: str, i: int) -> tuple[list[str], int]:
    """Options in a seeded random order (per video and question) and the gold's new index.
    The published order puts the answer first in 1,484 of 1,489 questions."""
    import random
    opts = list(q["options"])
    random.Random(f"{MCQ_SEED}:{vid}:{i}").shuffle(opts)
    return opts, opts.index(q["answer"])


def mcq_prompts(ids: list[str], cfg: dict, dcfg: dict, encoder) -> list[dict]:
    """One baseline-retrieval MCQ prompt per question of every prepared video."""
    qa, out = load_qa(), []
    k = dcfg["top_k"]
    for vid in ids:
        index_dir = PROCESSED / vid / "index"
        if not index_dir.exists():
            continue
        index = Index.load(index_dir)
        mcqs = [q for q in qa.get(vid, []) if q["kind"] == "mcq"]
        for i, q in enumerate(mcqs):
            res = [RetrievedUnit(unit=u, score=s, origin="seed")
                   for u, s in retrieve_scored(q["question"], index, encoder=encoder, top_k=k,
                                               candidates=cfg["retrieve"]["candidates"], rrf_k=cfg["retrieve"]["rrf_k"])]
            opts, gold = shuffled(q, vid, i)
            listing = "\n".join(f"{'ABCD'[j]}. {o}" for j, o in enumerate(opts))
            user = f"Evidence:\n{evidence_block(res)}\n\nQuestion: {q['question']}\n{listing}\n\nLetter:"
            out.append({"vid": vid, "i": i, "level": q["level"], "user": user, "options": opts, "gold": "ABCD"[gold]})
    return out


def mcq_shuffled(ids: list[str], cfg: dict, dcfg: dict, out: Path, *, max_cost: float, dry_run: bool,
                 subset_n: int = 100, repeats: int = 3) -> int:
    """MCQ with shuffled options, baseline retrieval, one run; a seeded subset of `subset_n`
    questions is repeated `repeats` times for run-to-run agreement."""
    import random
    import statistics as st
    import tiktoken
    from linkrag.eval import lectqa_metrics as LM
    llm = {**cfg["models"]["llm"], "max_tokens": 32}           # one letter; 8 cut off a reply (API 400)
    price = price_for(str(llm["model"]), cfg["models"].get("pricing"))
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    items = mcq_prompts(ids, cfg, dcfg, encoder)
    subset = set(random.Random(MCQ_SEED + ":subset").sample(range(len(items)), min(subset_n, len(items))))
    enc = tiktoken.get_encoding("o200k_base")
    calls = len(items) + (repeats - 1) * len(subset)
    tok_in = sum(len(enc.encode(MCQ_SYSTEM + it["user"])) + 8 for it in items)
    tok_in += (repeats - 1) * sum(len(enc.encode(MCQ_SYSTEM + items[j]["user"])) + 8 for j in subset)
    est = tok_in / 1e6 * price["input_per_m"] + calls * 2 / 1e6 * price["output_per_m"]
    print(f"{len(items)} MCQs on {len({it['vid'] for it in items})} videos · {calls} calls · ~{tok_in:,} input tokens · "
          f"estimated ${est:.3f} (cap ${max_cost:.2f})")
    if dry_run:
        return 0
    if est > max_cost:
        raise SystemExit(f"estimated ${est:.3f} exceeds --max-cost {max_cost}; nothing sent")
    from concurrent.futures import ThreadPoolExecutor
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / f"{llm['model']}-mcq-shuffled",
                                max_cost_usd=max_cost, price=price)

    def answer(j):
        it = items[j]
        it["preds"] = []
        for _ in range(repeats if j in subset else 1):
            c = mcq_choice(complete(MCQ_SYSTEM, it["user"]), it["options"])
            it["preds"].append(None if c is None else "ABCD"[c])

    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, _ in enumerate(ex.map(answer, range(len(items))), 1):
            if n % 200 == 0:
                print(f"  {n}/{len(items)} answered · ${usage_cost(complete.usage, price) or 0:.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps({k: v for k, v in it.items() if k != "user"}) for it in items) + "\n")
    gold_pos = collections.Counter(it["gold"] for it in items)
    L = ["# LectQA-Vid — MCQ with shuffled options", "",
         f"{len(items)} MCQs · {len({it['vid'] for it in items})} videos · answerer `{llm['model']}` temperature "
         f"{llm.get('temperature')} · baseline retrieval (RRF dense + BM25, top-{dcfg['top_k']}) · options shuffled "
         f"with seed `{MCQ_SEED}` (per video and question) · one run.", "",
         "Gold position after shuffling: " + ", ".join(f"{k} {v}" for k, v in sorted(gold_pos.items()))
         + " (before: A in 1,484 of 1,489).", "",
         "| level | n | ACC | Precision | Recall | F1 | theirs ACC (Table 5) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        sel = [it for it in items if level == "overall" or it["level"] == level]
        s = LM.score_mcq([it["preds"][0] for it in sel], [it["gold"] for it in sel])
        L.append(f"| {level} | {s['n']} | {100*s['accuracy']:.2f} | {100*s['precision']:.2f} | {100*s['recall']:.2f} | "
                 f"{100*s['f1']:.2f} | {THEIRS['mcq'][level]:.2f} |")
    longest = {id(it): longest_option(it["options"]) for it in items}
    L += ["", "**Answerable from the options alone.** A no-video answerer that picks the unique longest option "
          "scores " + ", ".join(
              f"{lv} {100 * st.mean(float(longest[id(it)] == it['gold']) for it in items if lv == 'overall' or it['level'] == lv):.1f}"
              for lv in (*LEVELS, "overall"))
          + f" on these {len(items)} questions (audit (a)). The answerer picked the longest option for "
          f"{sum(it['preds'][0] == longest[id(it)] for it in items)}/{len(items)}. The per-difficulty accuracy above "
          "follows the longest-option share, so this accuracy does not measure retrieval either."]
    sub_items = [items[j] for j in sorted(subset)]
    agree = sum(len(set(it["preds"])) == 1 for it in sub_items)
    accs = [st.mean(float(it["preds"][r] == it["gold"]) for it in sub_items) for r in range(repeats)]
    L += ["", f"**Run-to-run agreement** on a seeded {len(sub_items)}-question subset, {repeats} repeats: identical "
          f"answer in all repeats for {agree}/{len(sub_items)}; subset accuracy per repeat "
          + ", ".join(f"{100*a:.1f}" for a in accs) + f" (mean ± std {100*st.mean(accs):.1f} ± {100*st.pstdev(accs):.1f})."]
    L += [""] + record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid MCQ shuffled",
                           [(str(llm["model"]), complete.usage)], cfg["models"].get("pricing"))
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


# ----------------------------------------------------------------- open-ended: T1 replica vs iterative vs full context

OPEN_SEED = "lectqa-open-subset-20260923"
OPEN_MODES = ("baseline", "iterative", "full_context")
# T1 Table 3 values; M and L (re-rank pool, context size) and the merge gap are not given
# in the paper -- ours: M < K as §4.4.3 requires, L = the top_k every other mode here gets.
T1 = {"theta_sem": 0.75, "theta_t": 2.0, "delta_gap": 5.0, "delta_dup": 0.90, "theta_r": 0.60, "dt": 3.0,
      "K": 10, "M": 8, "L": 4}
T1_EMBEDDER = "BAAI/bge-base-en-v1.5"                 # "BGE base", d = 768, mean-pooled (eqs. 14-15)
T1_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # "fine-tuned on QA relevance pairs", r in [0, 1]; unnamed
EVIDENCE = PROCESSED / "open_evidence.jsonl"          # item-4 input: the units each subset answer saw


def t1_encoder(device: str):
    from sentence_transformers import SentenceTransformer, models
    tr = models.Transformer(T1_EMBEDDER)
    m = SentenceTransformer(modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean"),
                                     models.Normalize()], device=device)
    return lambda texts: m.encode(list(texts), convert_to_numpy=True)


def t1_chunks(vid: str, index: Index, enc) -> tuple[list[EvidenceUnit], np.ndarray]:
    """T1 §4.2 on our inputs: whisper words re-split into sentences; a frame-OCR unit
    overlapping speech by more than theta_t is suppressed (audio priority, eq. 10);
    near-duplicates (cosine > delta_dup to a kept item) dropped; adjacent items merged
    while cosine > theta_sem and gap < delta_gap (eqs. 11-12). Text normalisation and
    boundary refinement (§4.2.1-2) are not replicated: whisper-small sentences are short."""
    audio = sorted((u for u in index.units if u.modality == "audio"), key=lambda u: u.location.start_s)
    items, cur = [], []
    for w in (w for u in audio for w in u.metadata.get("words", [])):
        cur.append(w)
        if str(w[2]).strip().endswith((".", "?", "!")):
            items.append(cur)
            cur = []
    items = [(" ".join(str(w[2]).strip() for w in s), float(s[0][0]), float(s[-1][1]), "audio") for s in items + [cur] if s]
    speech = list(items)
    for f in index.units:
        if f.modality == "figure" and f.content.strip():
            a, b = f.location.start_s, f.location.end_s
            if max((min(b, e) - max(a, s) for _, s, e, _ in speech), default=0.0) <= T1["theta_t"]:
                items.append((" ".join(f.content.split()), a, b, "figure"))
    items.sort(key=lambda x: x[1])
    vec = np.asarray(enc([x[0] for x in items]), dtype=np.float32)
    keep: list[int] = []
    for i in range(len(items)):
        if not keep or float(np.max(vec[keep] @ vec[i])) <= T1["delta_dup"]:
            keep.append(i)
    groups = [[keep[0]]] if keep else []
    for a, b in zip(keep, keep[1:]):
        if float(vec[a] @ vec[b]) > T1["theta_sem"] and items[b][1] - items[a][2] < T1["delta_gap"]:
            groups[-1].append(b)
        else:
            groups.append([b])
    units = [EvidenceUnit(id=f"{vid}:c{n}", modality="audio" if any(items[i][3] == "audio" for i in g) else "figure",
                          content=" ".join(items[i][0] for i in g), source_file=audio[0].source_file,
                          location=Location(start_s=items[g[0]][1], end_s=max(items[i][2] for i in g)))
             for n, g in enumerate(groups)]
    return units, np.asarray(enc([u.content for u in units]), dtype=np.float32)


def t1_retrieve(q: str, gi: tuple[float, float] | None, chunks: list[EvidenceUnit], vecs: np.ndarray, enc,
                ce) -> list[EvidenceUnit]:
    """T1 §4.4: top-K cosine above theta_r (eqs. 17, 21); the temporal filter of eq. 22 on the
    question's annotated interval +- dt when it is usable; merge of temporally consecutive
    results with gap < delta_gap (eq. 23); cross-encoder over the top-M (eqs. 24-25); the top-L
    in temporal order (eqs. 26-27). Eq. 22 reads the gold timestamps: an oracle."""
    sims = vecs @ np.asarray(enc([q]), dtype=np.float32)[0]
    top = [i for i in np.argsort(-sims)[:T1["K"]] if sims[i] > T1["theta_r"]]
    if gi is not None:
        lo, hi = gi[0] - T1["dt"], gi[1] + T1["dt"]
        top = [i for i in top if chunks[i].location.start_s <= hi and chunks[i].location.end_s >= lo]
    top.sort(key=lambda i: chunks[i].location.start_s)
    groups: list[list[int]] = []
    for i in top:
        if groups and chunks[i].location.start_s - chunks[groups[-1][-1]].location.end_s < T1["delta_gap"]:
            groups[-1].append(i)
        else:
            groups.append([i])
    merged = [(max(sims[i] for i in g),
               EvidenceUnit(id=f"{chunks[g[0]].id}" + (f"-{chunks[g[-1]].id.rsplit(':', 1)[1]}" if len(g) > 1 else ""),
                            modality=chunks[g[0]].modality, content=" ".join(chunks[i].content for i in g),
                            source_file=chunks[g[0]].source_file,
                            location=Location(start_s=chunks[g[0]].location.start_s,
                                              end_s=max(chunks[i].location.end_s for i in g))))
              for g in groups]
    pool = [u for _, u in sorted(merged, key=lambda x: -x[0])[:T1["M"]]]
    if not pool:
        return []
    r = ce.predict([(q, u.content) for u in pool])
    best = [pool[i] for i in np.argsort(-np.asarray(r))[:T1["L"]]]
    return sorted(best, key=lambda u: u.location.start_s)


def open_items(ids: list[str]) -> tuple[list[dict], set[int]]:
    """Every open-ended question of the prepared videos, and a seeded stratified subset
    (100 per difficulty)."""
    import random
    qa, items = load_qa(), []
    for vid in ids:
        if (PROCESSED / vid / "index").exists():
            items += [{"vid": vid, "qi": i, "level": q["level"], "question": q["question"], "ref": q["answer"], "q": q}
                      for i, q in enumerate(x for x in qa.get(vid, []) if x["kind"] == "open")]
    rng = random.Random(OPEN_SEED)
    subset = set()
    for level in LEVELS:
        subset |= set(rng.sample([j for j, it in enumerate(items) if it["level"] == level], 100))
    return items, subset


def open_modes(ids: list[str], cfg: dict, dcfg: dict, out: Path, *, max_cost: float, dry_run: bool,
               repeats: int = 3, rest: int | None = None) -> int:
    """Open-ended answers under T1's protocol and metrics: baseline (T1 replica), iterative
    (ours), full_context (no retrieval). The subset runs `repeats` times; the rest once,
    in seeded order, `rest` questions of it (all by default)."""
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import tiktoken
    from sentence_transformers import CrossEncoder
    from linkrag.generate.answer import JSON_SYSTEM, answer_json, build_prompt
    from linkrag.retrieve.iterative import retrieve_iterative
    llm = {**cfg["models"]["llm"], "max_tokens": 400}
    price = price_for(str(llm["model"]), cfg["models"].get("pricing"))
    items, subset = open_items(ids)
    others = [j for j in range(len(items)) if j not in subset]
    random.Random(OPEN_SEED + ":rest").shuffle(others)
    others = others[:rest] if rest is not None else others
    run_idx = sorted(subset) + others
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    t1enc, ce = t1_encoder(cfg["device"]), CrossEncoder(T1_RERANKER, device=cfg["device"])
    indexes, chunks = {}, {}
    for vid in sorted({items[j]["vid"] for j in run_idx}):
        indexes[vid] = Index.load(PROCESSED / vid / "index")
        chunks[vid] = t1_chunks(vid, indexes[vid], t1enc)
    fixed: dict[tuple[int, str], list[EvidenceUnit]] = {}
    oracle = {}
    for j in run_idx:
        it = items[j]
        gi, _ = gold_interval(it["vid"], it["q"])
        oracle[j] = gi is not None
        fixed[(j, "baseline")] = t1_retrieve(it["question"], gi, *chunks[it["vid"]], t1enc, ce)
        fixed[(j, "full_context")] = sorted((u for u in indexes[it["vid"]].units if u.modality == "audio"),
                                            key=lambda u: u.location.start_s)
    # repeat-major: a cost cap reached mid-run cuts the last repeat, not whole questions
    tasks = ([(j, m, r) for r in range(repeats) for j in sorted(subset) for m in OPEN_MODES]
             + [(j, m, 0) for j in others for m in OPEN_MODES])

    # cost estimate: exact prompt tokens for the fixed-evidence modes; the iterative prompt is
    # taken as the baseline's plus one follow-up call over 8 units; 110 output tokens per answer
    enc = tiktoken.get_encoding("o200k_base")
    ptok = lambda units, q: len(enc.encode(JSON_SYSTEM + build_prompt(q, units))) + 8
    tok_in = tok_out = 0
    for j, m, _ in tasks:
        q = items[j]["question"]
        if m == "iterative":
            tok_in += ptok(fixed[(j, "baseline")] or fixed[(j, "full_context")][:T1["L"]], q) + 600
            tok_out += 110 + 25
        else:
            tok_in += ptok(fixed[(j, m)], q) if fixed[(j, m)] else 0
            tok_out += 110 if fixed[(j, m)] else 0
    est = tok_in / 1e6 * price["input_per_m"] + tok_out / 1e6 * price["output_per_m"]
    empty = sum(1 for j in run_idx if not fixed[(j, "baseline")])
    print(f"{len(items)} open-ended on {len({it['vid'] for it in items})} videos · subset {len(subset)} x {repeats} "
          f"+ rest {len(others)} x 1 · {len(tasks)} answers · ~{tok_in:,} in / ~{tok_out:,} out · estimated ${est:.2f} "
          f"(cap ${max_cost:.2f}) · baseline: oracle filter on {sum(oracle[j] for j in run_idx)}/{len(run_idx)}, "
          f"empty context {empty}")
    if dry_run:
        return 0
    if est > max_cost:
        raise SystemExit(f"estimated ${est:.2f} exceeds --max-cost {max_cost}; nothing sent (lower --rest)")

    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / f"{llm['model']}-open",
                                max_cost_usd=max_cost, price=price)
    lock = threading.Lock()

    def locked(texts):
        with lock:
            return encoder(texts)

    def one(task):
        try:
            return answer_one(task)
        except RuntimeError as exc:                   # the cost cap: report what ran, never exceed it
            if "cost cap" not in str(exc):
                raise
            return None, None

    def answer_one(task):
        j, mode, rep = task
        it = items[j]
        units = fixed.get((j, mode))
        if mode == "iterative":
            icfg = cfg["retrieve"]["iterative"]
            units = [r.unit for r in retrieve_iterative(
                it["question"], indexes[it["vid"]], encoder=locked, complete=complete, rounds=icfg["rounds"],
                k_per_round=icfg["k_per_round"], k_final=T1["L"], candidates=cfg["retrieve"]["candidates"],
                rrf_k=cfg["retrieve"]["rrf_k"]).units]
        ans = answer_json(it["question"], units, mode="baseline", complete=complete)
        row = {"vid": it["vid"], "qi": j, "level": it["level"], "mode": mode, "rep": rep, "subset": j in subset,
               "oracle": oracle[j] and mode == "baseline", "answer": ans.answer, "ref": it["ref"],
               "malformed": ans.malformed, "claims": [{"claim": c.claim, "unit_ids": c.unit_ids} for c in ans.claims],
               "evidence": [u.id for u in units]}
        ev = None
        if j in subset and rep == 0:
            ev = {"qi": j, "mode": mode, "units": [{"id": u.id, "modality": u.modality, "content": u.content,
                                                    "source_file": u.source_file, "start_s": u.location.start_s,
                                                    "end_s": u.location.end_s} for u in units]}
        return row, ev

    rows, evs = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, (row, ev) in enumerate(ex.map(one, tasks), 1):
            if row:
                rows.append(row)
            if ev:
                evs.append(ev)
            if n % 300 == 0:
                print(f"  {n}/{len(tasks)} answered · ${usage_cost(complete.usage, price) or 0:.3f}")
    EVIDENCE.write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return open_report(rows, items, subset, cfg, llm, repeats, out,
                       record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid open-ended baseline/iterative/full_context",
                                  [(str(llm["model"]), complete.usage)], cfg["models"].get("pricing")))


def faithfulness(cfg: dict, src: Path, out: Path, *, max_cost: float, dry_run: bool, runs: int) -> int:
    """Claim-level verification (`linkrag.generate.verify`, judge `eval.judge`) of the subset's
    repeat-0 answers, every mode: each claim against the units it cites."""
    import tiktoken
    from concurrent.futures import ThreadPoolExecutor
    from linkrag.eval.verify_gold import ENTAIL_PROMPT, JUDGE_SYSTEM
    from linkrag.generate.answer import Answer, Claim
    from linkrag.generate.verify import hallucination_rate, verify_answer
    rows = [r for r in map(json.loads, src.read_text().splitlines()) if r["subset"] and r["rep"] == 0]
    ev = {}
    for e in map(json.loads, EVIDENCE.read_text().splitlines()):
        ev[(e["qi"], e["mode"])] = [EvidenceUnit(id=u["id"], modality=u["modality"], content=u["content"],
                                                 source_file=u["source_file"],
                                                 location=Location(start_s=u["start_s"], end_s=u["end_s"]))
                                    for u in e["units"]]
    judge = cfg["eval"]["judge"]
    price = price_for(str(judge["model"]), cfg["models"].get("pricing"))
    enc = tiktoken.get_encoding("o200k_base")
    lo = hi = 0                                        # input tokens: first cited unit only / every cited unit
    n_claims = 0
    for r in rows:
        by_id = {u.id: u for u in ev[(r["qi"], r["mode"])]}
        for c in r["claims"]:
            n_claims += 1
            toks = [len(enc.encode(JUDGE_SYSTEM + ENTAIL_PROMPT.format(question=c["claim"], expected=c["claim"],
                                                                      text=by_id[i].content))) + 8
                    for i in c["unit_ids"] if i in by_id]
            lo, hi = lo + (toks[0] if toks else 0), hi + sum(toks)
    cost = lambda tin, calls: runs * (tin / 1e6 * price["input_per_m"] + calls * 120 / 1e6 * price["output_per_m"])
    est_lo, est_hi = cost(lo, n_claims), cost(hi, n_claims * 2)
    print(f"{len(rows)} answers · {n_claims} claims · judge `{judge['model']}` x {runs} runs · estimated "
          f"${est_lo:.2f}-${est_hi:.2f} (cap ${max_cost:.2f})")
    if dry_run:
        return 0
    if est_hi > max_cost:
        raise SystemExit(f"upper estimate ${est_hi:.2f} exceeds --max-cost {max_cost}; nothing sent (lower --runs)")
    complete = cached_completer(http_completer(judge), PROCESSED / "llm_cache" / f"{judge['model']}-faith",
                                max_cost_usd=max_cost, price=price)

    def check(r):
        ans = Answer(answer=r["answer"], claims=[Claim(**c) for c in r["claims"]], raw="")
        return {**{k: r[k] for k in ("vid", "qi", "level", "mode")},
                **verify_answer(ans, ev[(r["qi"], r["mode"])], complete, runs=runs, workers=4)}

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(check, rows))
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(x) for x in res) + "\n")
    answerer = str(cfg["models"]["llm"]["model"])
    L = ["# LectQA-Vid — faithfulness per mode", "",
         f"The {len(res)} repeat-0 answers of the open-ended subset (`lectqa_open_modes.md`: 100 questions per "
         f"difficulty, seed `{OPEN_SEED}`, {len({x['vid'] for x in res})} videos) · answerer `{answerer}` · judge "
         f"`{judge['model']}` temperature {judge.get('temperature')}, {runs} run(s) per check"
         + (", majority" if runs > 1 else "") + " · `linkrag.generate.verify.verify_answer`: each claim against "
         "the text of the units it cites; the first entailing unit wins, with a quotable span required.", "",
         "**supported**: a cited unit entails the claim. **weak**: the claim cites no unit of the evidence. "
         "**unsupported**: no cited unit entails it. **Hallucination rate** = unsupported / claims, pooled over "
         "answers (`hallucination_rate`). Supported means the claim is stated in the cited evidence. It does "
         "not mean the answer is correct.", "",
         "| level | mode | answers | claims | supported | weak | unsupported | hallucination rate | answers with no claim |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for mode in OPEN_MODES:
            sel = [x for x in res if x["mode"] == mode and (level == "overall" or x["level"] == level)]
            if sel:
                claims = sum(len(x["claims"]) for x in sel)
                L.append(f"| {level} | {mode} | {len(sel)} | {claims} | {sum(x['supported'] for x in sel)} | "
                         f"{sum(x['weak'] for x in sel)} | {sum(x['unsupported'] for x in sel)} | "
                         f"{100 * hallucination_rate(sel):.1f} % | {sum(not x['claims'] for x in sel)} |")
    tot = sum(len(x["claims"]) for x in res)
    assert tot == sum(x[k] for x in res for k in ("supported", "weak", "unsupported")), "verdicts must sum to claims"
    L += [""] + record_run("scripts/adapters/lectqa_vid.py", "lectqa_vid faithfulness per mode",
                           [(str(judge["model"]), complete.usage)], cfg["models"].get("pricing"))
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


METRICS = (("f1", "F1"), ("sim", "Sim"), ("bleu", "BLEU"), ("meteor", "METEOR"), ("rouge1", "R1"))


def open_report(rows: list[dict], items: list[dict], subset: set[int], cfg: dict, llm: dict, repeats: int,
                out: Path, footer: list[str]) -> int:
    import statistics as st
    from linkrag.eval import lectqa_metrics as LM
    score = lambda sel: LM.score_open([r["answer"] for r in sel], [r["ref"] for r in sel], device=cfg["device"])
    sub_vids = len({items[j]["vid"] for j in subset})
    all_vids = len({r["vid"] for r in rows})
    label = (f"answerer `{llm['model']}` temperature {llm.get('temperature')} (JSON answer with claims, the same "
             f"prompt for every mode) · metrics T1 eqs. 29-35 (`linkrag.eval.lectqa_metrics`), in %")
    L = ["# LectQA-Vid — open-ended: T1 replica vs iterative vs full context", "",
         "Three ways to give the same answerer its context, scored with T1's own metric suite:", "",
         f"- **baseline**: T1's pipeline (§4.2-4.4) replicated. Chunks: whisper sentences merged while cosine > "
         f"{T1['theta_sem']} and gap < {T1['delta_gap']} s, near-duplicates (> {T1['delta_dup']}) dropped, frame OCR "
         f"overlapping speech by > {T1['theta_t']} s suppressed; embedder `{T1_EMBEDDER}` mean-pooled; top-{T1['K']} "
         f"above cosine {T1['theta_r']}; temporal filter (eq. 22) on the question's annotated interval ± {T1['dt']} s; "
         f"merge of consecutive results with gap < {T1['delta_gap']} s; cross-encoder `{T1_RERANKER}` over the top-"
         f"{T1['M']}; top-{T1['L']} in temporal order. **Eq. 22 reads the gold timestamps (an oracle).** It is applied "
         "only where the stamp is usable (`gold_interval`); elsewhere the pipeline runs without it. Both are "
         "reported separately below.",
         f"- **iterative**: ours: RRF (bge-m3 dense + BM25) over our sentence-packed transcript and frame-OCR "
         f"units, one LLM follow-up query, top-{T1['L']}. No timestamps are read.",
         "- **full_context**: no retrieval; every transcript unit of the video, in time order. No timestamps "
         "and no frame OCR.", "",
         "T1 leaves open: M, L and the merge gap (ours above); the cross-encoder (ours: MS MARCO MiniLM); one "
         "FAISS index over all videos then a video-id filter (ours: each video's own chunks, i.e. that filter "
         "without other videos competing for the top-K); Gemini captions (ours: tesseract OCR, mostly "
         "suppressed by their θt rule).", ""]
    rest = len(items) - len(subset)
    if not any(not r["subset"] for r in rows):
        L += [f"**Scope.** The single run on the other {rest} open-ended questions was not made: the dry run "
              "estimated $7.43 for subset + rest against the item's $3.50 cap, so only the subset ran.", ""]
    tables = [(f"## Subset: {len(subset)} questions (100 per difficulty, seed `{OPEN_SEED}`), {sub_vids} videos, "
               f"{repeats} repeats, mean ± std over repeats", [r for r in rows if r["subset"]], repeats)]
    if any(not r["subset"] for r in rows):
        tables.append((f"## All run questions, one run (subset repeat 0 + the rest), {all_vids} videos",
                       [r for r in rows if r["rep"] == 0], 1))
    for title, sel_rows, reps in tables:
        L += [title, "", label + ".", "",
              "| level | mode | n | " + " | ".join(h for _, h in METRICS) + " | abstained |",
              "|---|---|---:|" + "---:|" * (len(METRICS) + 1)]
        for level in (*LEVELS, "overall"):
            for mode in OPEN_MODES:
                sel = [r for r in sel_rows if r["mode"] == mode and (level == "overall" or r["level"] == level)]
                if not sel:
                    continue
                per = [score([r for r in sel if r["rep"] == k]) for k in range(reps)]
                cell = lambda key: (f"{100*st.mean(p[key] for p in per):.2f} ± {100*st.pstdev(p[key] for p in per):.2f}"
                                    if reps > 1 else f"{100*per[0][key]:.2f}")
                abst = sum("not found in the provided material" in r["answer"].lower() or
                           r["answer"].startswith("No evidence was retrieved") for r in sel if r["rep"] == 0)
                L.append(f"| {level} | {mode} | {per[0]['n']} | " + " | ".join(cell(k) for k, _ in METRICS)
                         + f" | {abst} |")
            th = THEIRS["open"][level]
            L.append(f"| {level} | **T1 Table 4** | — | " + " | ".join(f"{th[k]:.2f}" for k, _ in METRICS) + " | — |")
        L.append("")
    # paired over questions: per-question F1 averaged over repeats, then a seeded bootstrap of the mean difference
    f1 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["subset"]:
            f1[(r["qi"], r["level"])][r["mode"]].append(LM.token_prf(r["answer"], r["ref"])[2])
    L += ["## Paired differences (subset, token F1 points, 95 % bootstrap CI over questions)", "",
          "Each question's F1 averaged over its repeats; the mean paired difference and a 10,000-resample "
          "bootstrap over questions (numpy seed 20260923). This is the question-sampling uncertainty; the ± "
          "above is run-to-run noise only.", "", "| difference | " + " | ".join((*LEVELS, "overall")) + " |",
          "|---|" + "---:|" * (len(LEVELS) + 1)]
    for a, b in (("full_context", "iterative"), ("full_context", "baseline"), ("iterative", "baseline")):
        cells = []
        for level in (*LEVELS, "overall"):
            d = np.array([100 * (st.mean(m[a]) - st.mean(m[b])) for (_, lv), m in f1.items()
                          if level == "overall" or lv == level])
            boots = np.random.default_rng(20260923).choice(d, (10000, len(d))).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cells.append(f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
        L.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    L.append("")
    one = [r for r in rows if r["rep"] == 0]
    L += ["## Baseline with and without its oracle filter (repeat 0, subset)", "", label + ".", "",
          "The two groups are different questions, so the other modes' F1 on the same groups is the control: "
          "a gap they share is the questions, not the filter.", "",
          "| questions | n | baseline F1 | baseline Sim | baseline empty context | iterative F1 | full_context F1 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, keep in (("eq. 22 applied (usable gold stamp)", True), ("eq. 22 not applicable", False)):
        qs = {r["qi"] for r in one if r["mode"] == "baseline" and r["oracle"] == keep}
        by = {m: [r for r in one if r["mode"] == m and r["qi"] in qs] for m in OPEN_MODES}
        if qs:
            s, si, sf = score(by["baseline"]), score(by["iterative"]), score(by["full_context"])
            L.append(f"| {name} | {len(qs)} | {100*s['f1']:.2f} | {100*s['sim']:.2f} | "
                     f"{sum(not r['evidence'] for r in by['baseline'])} | {100*si['f1']:.2f} | {100*sf['f1']:.2f} |")
    L += ["", f"Malformed JSON replies (answer scored as the raw text): {sum(r['malformed'] for r in rows)} of {len(rows)}."
          f" Answers run: {len(rows)} (subset {sum(r['subset'] for r in rows)}, rest {sum(not r['subset'] for r in rows)})."]
    L += footer
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def open_rerender(ids: list[str], cfg: dict, out: Path, repeats: int = 3) -> int:
    """The report again from the stored answers: no LLM call, the original run's cost footer kept."""
    rows = [json.loads(x) for x in out.with_suffix(".jsonl").read_text().splitlines()]
    items, subset = open_items(ids)
    text = out.read_text()
    footer = text[text.index("\nLLM cost (this run):"):].rstrip("\n").split("\n")
    return open_report(rows, items, subset, cfg, cfg["models"]["llm"], repeats, out, footer)


# ----------------------------------------------------------------- benchmark audit

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


# ----------------------------------------------------------------- coverage

def coverage(out: Path) -> int:
    """Per-video acquisition status: source, obtained/dead (with the error), prepared."""
    status = json.loads((RAW / "fetch_status.json").read_text()) if (RAW / "fetch_status.json").exists() else {}
    qa = load_qa()
    rows, obtained, prepared = [], 0, 0
    for vid in video_ids(None, None):
        st_ = status.get(vid, {"status": "not attempted"})
        is_prep = (PROCESSED / vid / "index").exists()
        obtained += st_["status"] == "obtained"
        prepared += is_prep
        rows.append(f"| {vid} | {st_.get('source', '—')} | {st_['status']} | {'yes' if is_prep else 'no'} | "
                    f"{len(qa.get(vid, []))} | {st_.get('error', '')[:70]} |")
    dead = [v for v in video_ids(None, None) if status.get(v, {}).get("status") == "dead"]
    L = ["# LectQA-Vid — acquisition coverage", "",
         "Source: the Mendeley record (doi:10.17632/yt4nmz9mcv.1, CC BY 4.0) ships the QA pairs and YouTube "
         "links only (\"videos, keyframes and transcripts are not provided because of copyright issues\"), so "
         "every video comes from YouTube: audio m4a + 360p video-only mp4, fetched with yt-dlp. Transcripts are "
         "ours: faster-whisper, sentence-split, frozen on first run.", "",
         f"**Obtained {obtained}/100 · prepared (transcribed + indexed) {prepared}/100 · dead {len(dead)}**: "
         + (", ".join(dead) or "none"), "",
         "| video | source | status | prepared | QA pairs | error |", "|---|---|---|---|---:|---|", *rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))
    return 0


# ----------------------------------------------------------------- frame-derived slides

def build_frameslides(vid: str, cfg: dict, encoder) -> dict | None:
    """index_frameslides for one prepared video: its audio units + slide page/figure units
    derived from the frames, and the full linking layer (link_corpus) over them."""
    from linkrag.ingest.video_slides import slide_at, slide_units
    from linkrag.link.align import align_naive
    from linkrag.link.pipeline import link_corpus
    video = RAW / "videos" / f"{vid}.video.mp4"
    base = PROCESSED / vid / "index"
    if not video.exists() or not base.exists():
        return None
    out = PROCESSED / vid / "index_frameslides"
    cached = out / "frameslides_stats.json"          # built before: reuse (delete the dir to rebuild)
    if cached.exists() and (out / "links.jsonl").exists():
        return json.loads(cached.read_text())
    audio = sorted((u for u in Index.load(base).units if u.modality == "audio"), key=lambda u: u.location.start_s or 0.0)
    units, slides = slide_units(video, vid, PROCESSED / vid / "frameslides")
    pages = sorted((u for u in units if u.modality == "text"), key=lambda u: u.location.page)
    figures = [u for u in units if u.modality == "figure"]
    all_units = [*audio, *units]
    index = build_index(all_units, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                        device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"])
    index.save(out)
    manifest = build_manifest([RAW / "videos" / f"{vid}.m4a", video], all_units)
    write_manifest(manifest, out / MANIFEST_NAME)
    if not audio or not pages:
        return {"vid": vid, "slides": len(slides), "figures": len(figures), "links": {}, "aligned": 0, "n_truth": 0}
    run = link_corpus(audio, pages, pages, figures, encoder=encoder, cfg=cfg)
    save_links(run.links, out / "links.jsonl", manifest_hash=manifest.get("hash"))
    # free sanity check: the slide on screen at each audio segment's midpoint is known
    truth = [slide_at(slides, ((u.location.start_s or 0) + (u.location.end_s or 0)) / 2) for u in audio]
    naive = align_naive(run.alignment.similarity)
    ok = [(t, p, nv) for t, p, nv in zip(truth, run.alignment.path, naive) if t is not None]
    stats = {"vid": vid, "slides": len(slides), "figures": len(figures), "audio": len(audio),
             "links": dict(collections.Counter(l.link_type for l in run.links)), "n_links": len(run.links),
             "gate": run.gate, "n_truth": len(ok), "aligned": sum(p == t for t, p, _ in ok),
             "naive": sum(nv == t for t, _, nv in ok)}
    cached.write_text(json.dumps(stats, default=float))
    return stats


def frameslides(ids: list[str], cfg: dict, dcfg: dict, out: Path) -> int:
    """Build index_frameslides for every prepared video, then compare localisation with and
    without frame-derived slides (retrieval only, no LLM: baseline and linkrag modes)."""
    import statistics as st
    encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
    encoder([""])
    stats = []
    for vid in ids:
        s = build_frameslides(vid, cfg, encoder)
        if s:
            stats.append(s)
            print(f"frameslides {vid}: {s['slides']} slides, {s['figures']} figures, links {s['links']}, "
                  f"aligned {s['aligned']}/{s['n_truth']}")
    done = [s["vid"] for s in stats]
    modes = ("baseline", "linkrag")
    without, info = localise(done, cfg, dcfg, index_name="index", label="without", modes=modes)
    with_, _ = localise(done, cfg, dcfg, index_name="index_frameslides", label="with", modes=modes, link_types=None)
    gold = info["gold"]

    link_tot = collections.Counter()
    for s in stats:
        link_tot.update(s["links"])
    n_links = sum(s.get("n_links", 0) for s in stats)
    assert sum(link_tot.values()) == n_links, "per-type link counts must sum to the total (measurement rule 2)"
    n_truth, aligned, naive = (sum(s[k] for s in stats) for k in ("n_truth", "aligned", "naive"))
    gated_out = sum(1 for s in stats if s.get("gate") and not s["gate"].get("related"))
    L = ["# LectQA-Vid — frame-derived slide units", "",
         f"{len(done)} prepared videos · slides from frames at 1 fps (dHash segmentation, transitions merged, "
         f"revisits deduplicated), OCR tesseract, figures by the deck clustering rules on raster frames "
         f"(`linkrag.ingest.video_slides`) · links from `linkrag.link.pipeline.link_corpus` (the pilot01 path). "
         f"**Retrieval only, no LLM, $0.** Iterative modes need a follow-up LLM call and are not run.", "",
         "*Without* = the prepared index: audio + one OCR unit per changed frame, joined by temporal "
         "co-occurrence links (T1's own signal). *With* = the same audio units + frame-derived slide page and "
         "figure units, joined by the linking layer (alignment, figure_text, same_slide, deictic); no temporal "
         "links, so the alignment has to find the slides from text. A revisited slide counts a hit on any of "
         "its on-screen intervals.", "",
         "## Slides, figures and links", "",
         f"- slides {sum(s['slides'] for s in stats)}, figure units {sum(s['figures'] for s in stats)}, "
         f"audio units {sum(s.get('audio', 0) for s in stats)}",
         f"- relatedness gate rejected the audio↔slide alignment on {gated_out} of {len(stats)} videos", "",
         "| link type | count |", "|---|---:|", *[f"| {k} | {v} |" for k, v in sorted(link_tot.items())],
         f"| **total** | {n_links} |", "",
         "## Alignment against the known on-screen interval (free sanity check)", "",
         f"Audio segments whose midpoint falls inside a slide's interval: n = {n_truth}. "
         f"DP alignment (config `link.align`) picks the on-screen slide for **{aligned}/{n_truth} = "
         f"{aligned / max(n_truth, 1):.1%}**; naive argmax on the same matrix {naive}/{n_truth} = "
         f"{naive / max(n_truth, 1):.1%}. (Their alignment component is T2, MaViLS; this is not their metric.)", "",
         "## Modality mix of the retrieved top-8 (rerank none; mean units per set)", "",
         "| variant | mode | audio | text (slide) | figure |", "|---|---|---:|---:|---:|"]
    for label, rows in (("without", without), ("with", with_)):
        for mode in modes:
            xs = [r["mods"] for r in rows if r["mode"] == mode and r["rerank"] == "none"]
            if xs:
                L.append(f"| {label} | {mode} | {st.mean(x.get('audio', 0) for x in xs):.2f} | "
                         f"{st.mean(x.get('text', 0) for x in xs):.2f} | {st.mean(x.get('figure', 0) for x in xs):.2f} |")
    total_q = sum(gold.values())
    L += ["", "## Localisation (hit@k: a retrieved unit overlaps the gold interval)", "",
          f"Questions with a usable gold interval: **{gold.get('ok', 0)} of {total_q}**. Excluded: "
          + ", ".join(f"{k} {v}" for k, v in sorted(gold.items()) if k != "ok") + ". The published "
          "annotations mix timestamp conventions (HH:MM:SS, SS:cc, SSS:cc, M:SSS:cc) whose meaning differs "
          "between videos, and some stamps end past the fetched video, so only unambiguous, in-range "
          "intervals are scored (`strict_seconds`, `gold_interval`).", ""]
    L += localisation_table(without, "without frame-derived slides") + [""]
    L += localisation_table(with_, "with frame-derived slides") + [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps({"stats": stats, "without": without, "with": with_}, default=str))
    print("\n".join(L))
    return 0


# ----------------------------------------------------------------- v2: segmentation ablation

def rebuild_fixed_index(vid: str, cfg: dict, dcfg: dict) -> tuple[float, float]:
    """Re-segment the existing transcript (word timestamps live in the audio units'
    metadata) into fixed windows and build `index_fixed`; frames reused. Returns
    (mid-sentence boundary rate sentence-mode, fixed-mode)."""
    from linkrag.ingest.audio import Word, segment_words
    src = Index.load(PROCESSED / vid / "index")
    audio = sorted([u for u in src.units if u.modality == "audio"], key=lambda u: u.location.start_s)
    frames = [u for u in src.units if u.modality != "audio"]
    words = [Word(float(w[0]), float(w[1]), str(w[2])) for u in audio for w in u.metadata.get("words", [])]
    ends_sentence = lambda w: w.text.strip().endswith((".", "?", "!"))
    mid_sentence = lambda buckets: (sum(1 for b in buckets[:-1] if b and not ends_sentence(b[-1])) / max(len(buckets) - 1, 1))
    sent_buckets = [[Word(float(w[0]), float(w[1]), str(w[2])) for w in u.metadata.get("words", [])] for u in audio]
    fixed = segment_words(words, dcfg["audio_segment_seconds"])
    units = [EvidenceUnit(id=f"{vid}:x{i}", modality="audio", content=" ".join(w.text for w in b),
                          source_file=audio[0].source_file, location=Location(start_s=b[0].start, end_s=b[-1].end),
                          metadata={"words": [(w.start, w.end, w.text) for w in b]})
             for i, b in enumerate(fixed) if b]
    out_dir = PROCESSED / vid / "index_fixed"
    if not out_dir.exists():
        encoder = default_encoder(cfg["models"]["embedding"], cfg["device"], cfg["index"]["normalize_embeddings"])
        build_index(units + frames, encoder=encoder, embedding_model=cfg["models"]["embedding"],
                    device=cfg["device"], normalize=cfg["index"]["normalize_embeddings"]).save(out_dir)
    return mid_sentence(sent_buckets), mid_sentence(fixed)


def v2(ids: list[str], cfg: dict, dcfg: dict, out: Path, *, cache_only: bool = True, max_cost: float = 1.0) -> int:
    import statistics as st
    ids = [v for v in ids if (PROCESSED / v / "index").exists()]
    rows_sent, u1 = localise(ids, cfg, dcfg, label="sentence", cache_only=cache_only, max_cost=max_cost)
    rates = [rebuild_fixed_index(v, cfg, dcfg) for v in ids]
    rows_fixed, u2 = localise(ids, cfg, dcfg, index_name="index_fixed", label="fixed", cache_only=cache_only, max_cost=max_cost)
    n_q = len({(r["vid"], r["kind"], r["level"]) for r in rows_sent}) and len(rows_sent) // 8
    L = [f"# LectQA-Vid — second pass ({len(ids)}/100 videos, {n_q} QA pairs)", "",
         f"Subset: the {len(ids)} of the first 35 videos whose YouTube links were still available; 7 were not "
         "(`data/raw/lectqa_vid/fetch_failures.txt`). Every number below is on this 28/100 subset, one run. "
         f"Transcript whisper-{cfg['models']['whisper']}, frames every {dcfg['frame_interval_s']} s + tesseract OCR, "
         f"{dcfg['audio_segment_seconds']}-second segments, retrieval pool 20, k = 8 for localisation.", "",
         "**Caption for every `linkrag` / `linkrag_iter` row:** single-stream setting — one video, no separate deck or paper — so the only link "
         "available is temporal co-occurrence between a transcript segment and the frames on screen; **cross-file linking is inactive by "
         "construction**. These rows test additive expansion + the complementarity reranker over transcript and frame-OCR units, nothing more.", "",
         "## 1. Temporal localisation (primary metric; no answerer)", "",
         "**Their paper reports no localisation metric.** Its evaluation section lists token-level precision/recall/F1, "
         "BLEU, METEOR, ROUGE-1 and semantic similarity for open-ended questions and accuracy/precision/recall/F1 over "
         "option labels for MCQ (§5.2) — nothing about whether a retrieved segment matched the annotated interval, and "
         "it never states its top-K. **This table is therefore ours-on-their-data**, with their fixed-window "
         "configuration replicated as the `fixed` baseline row in §2 rather than a number copied from the paper.", "",
         "hit@k: any of the top-k retrieved units overlaps the question's gold interval `[timestamp_start, timestamp_end]` "
         "(the files mix `HH:MM:SS`, `MM:SS` and plain-seconds stamps; all three are parsed); "
         "mean best IoU: the best temporal IoU among the top 8. `iterative` / `linkrag_iter` need one LLM call per question "
         "for the follow-up query; **this run made no new LLM calls** — those cells are filled only where the call was already "
         f"cached, so their n is smaller (sentence pass: {u1['skipped_cells']} question-mode cells skipped; fixed pass: "
         f"{u2['skipped_cells']}). No answer is generated for this table.", ""]
    L += localisation_table(rows_sent, "Sentence-aware segmentation (`ingest.audio_segmentation: sentence`, the default) — per difficulty, all modes × rerank", WITH_GATE) + [""]
    conc = [r for r in rows_sent if r["rerank"] == "complementarity+gate" and r["gate"] == "concentrated"]
    L += [f"Modality-need gate: {len(conc)}/{len([r for r in rows_sent if r['rerank'] == 'complementarity+gate'])} "
          "query-mode cells were judged *concentrated* (alpha set to 0); the rest ran full complementarity.", ""]
    L += ["## 2. Segmentation ablation on their data", "",
          "Per difficulty, `baseline` / `rerank=none` (the row their fixed-window configuration corresponds to):", "",
          "| level | segmentation | hit@1 | hit@3 | hit@8 | mean best IoU |", "|---|---|---:|---:|---:|---:|"]
    for level in (*LEVELS, "overall"):
        for nm, rws in (("sentence", rows_sent), ("fixed", rows_fixed)):
            xs = [r for r in rws if r["mode"] == "baseline" and r["rerank"] == "none" and (level == "overall" or r["level"] == level)]
            if xs:
                m = lambda k: st.mean(r[k] for r in xs)
                L.append(f"| {level} | {nm} | {m('hit@1'):.1%} | {m('hit@3'):.1%} | {m('hit@8'):.1%} | {m('best_iou'):.3f} |")
    L += ["",
          "Same transcript words, same frames, same retrieval; only the cut points change. `fixed` = equal-time windows "
          f"({dcfg['audio_segment_seconds']} s), what the baseline papers do; `sentence` = split on terminal punctuation, then packed.", "",
          "| segmentation | mid-sentence boundary rate | hit@1 (overall, baseline/none) | hit@8 | mean best IoU |", "|---|---:|---:|---:|---:|"]
    for name_, rows, rate in (("sentence", rows_sent, st.mean(r[0] for r in rates)), ("fixed", rows_fixed, st.mean(r[1] for r in rates))):
        xs = [r for r in rows if r["mode"] == "baseline" and r["rerank"] == "none"]
        L.append(f"| {name_} | {rate:.1%} | {st.mean(r['hit@1'] for r in xs):.1%} | {st.mean(r['hit@8'] for r in xs):.1%} | {st.mean(r['best_iou'] for r in xs):.3f} |")
    L += [""] + localisation_table(rows_fixed, "Fixed windows (their configuration) — per difficulty, all modes × rerank", WITH_GATE) + [""]
    # 3. answer F1 vs their rows
    fr = Path("reports/lectqa_vid_first_run.jsonl")
    L += ["## 3. Answer token-F1 per difficulty vs their Table 4 (not directly comparable)", "",
          "Their paper does not name the answering LLM (\"an instruction-tuned LLM\", §4.4.4) and evaluates against LLaVA-1.6, so no "
          "matched-strength row can be built; ours is `gpt-5.4-2026-03-05`. **Answer-F1 is therefore not directly comparable — lead with "
          "localisation above.** Their semantic similarity used all-MiniLM-L6-v2; the first-run column used bge-m3 and is omitted here.", "",
          "| level | n | mode | token-F1 | ROUGE-1 | their F1 (Table 4) | their ROUGE-1 |", "|---|---:|---|---:|---:|---:|---:|"]
    if fr.exists():
        frs = [json.loads(l) for l in fr.read_text().splitlines()]
        for level in (*LEVELS, "overall"):
            for mode in ("baseline", "linkrag"):
                xs = [r for r in frs if r["kind"] == "open" and r["mode"] == mode and (level == "overall" or r["level"] == level)]
                if xs:
                    tf, tr = THEIRS["open"][level]["f1"], THEIRS["open"][level]["rouge1"]
                    L.append(f"| {level} | {len(xs)} | {mode} | {st.mean(r['f1'] for r in xs):.1%} | {st.mean(r['rouge1'] for r in xs):.1%} | {tf:.2f}% | {tr:.2f}% |")
    footer = record_run("scripts/adapters/lectqa_vid.py", f"lectqa v2 localisation ({len(ids)} videos)",
                        [(u1["model"], u1["llm_usage"]), (u2["model"], u2["llm_usage"])], cfg["models"].get("pricing"))
    L += footer
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps({**r, "segmentation": "sentence"}) for r in rows_sent)
                                         + "\n" + "\n".join(json.dumps({**r, "segmentation": "fixed"}) for r in rows_fixed) + "\n")
    print("\n".join(L[12:40]))
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "prepare", "run", "v2", "frameslides", "coverage", "audit", "mcq",
                                     "open", "faith"])
    ap.add_argument("--runs", type=int, default=3, help="faith: judge runs per check (majority)")
    ap.add_argument("--rest", type=int, default=None, help="open: how many non-subset questions (seeded order)")
    ap.add_argument("--report-only", action="store_true", help="open: re-render the report from the stored answers")
    ap.add_argument("--dry-run", action="store_true", help="mcq/open: build prompts and estimate cost only")
    ap.add_argument("--videos", type=int, default=None, help="first N video ids")
    ap.add_argument("--only", default=None, help="comma-separated video ids")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modes", default="baseline,linkrag")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="reports/lectqa_vid_first_run.md")
    ap.add_argument("--allow-new-calls", action="store_true", help="v2: let iterative modes call the LLM on cache misses")
    ap.add_argument("--metrics", choices=["ours", "theirs"], default="ours",
                    help="run: 'theirs' = T1's suite (token P/R/F1, BLEU, METEOR, ROUGE-1 recall, MiniLM sim, MCQ)")
    ap.add_argument("--cache-only", action="store_true", help="run: replay stored answers only; a miss is skipped")
    ap.add_argument("--answerer-model", default=None, help="override models.llm.model (e.g. the stored gpt-5.4)")
    ap.add_argument("--max-cost", type=float, required=True,
                    help="USD budget for this run (required; 0 = cache replays and free backends only)")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    if args.answerer_model:
        cfg["models"]["llm"]["model"] = args.answerer_model
    refuse_strong_in_batch(cfg, answerer_cache_only=args.cache_only)
    dcfg = cfg.get("datasets", {}).get("lectqa_vid", {"audio_segment_seconds": 15, "frame_interval_s": 5, "top_k": 4})
    ids = video_ids(args.videos, args.only)
    if args.step == "fetch":
        fetch(ids)
        return 0
    if args.step == "prepare":
        prepare(ids, cfg, dcfg)
        return 0
    if args.step == "mcq":
        return mcq_shuffled(ids, cfg, dcfg, Path("results/external/lectqa_mcq_shuffled.md"),
                            max_cost=args.max_cost, dry_run=args.dry_run)
    if args.step == "open" and args.report_only:
        return open_rerender(ids, cfg, Path("results/external/lectqa_open_modes.md"))
    if args.step == "open":
        return open_modes(ids, cfg, dcfg, Path("results/external/lectqa_open_modes.md"), max_cost=args.max_cost,
                          dry_run=args.dry_run, rest=args.rest)
    if args.step == "faith":
        return faithfulness(cfg, Path("results/external/lectqa_open_modes.jsonl"),
                            Path("results/external/lectqa_faithfulness.md"), max_cost=args.max_cost,
                            dry_run=args.dry_run, runs=args.runs)
    if args.step == "audit":
        return audit(Path("results/external/lectqa_audit.md"))
    if args.step == "coverage":
        return coverage(Path("results/external/lectqa_coverage.md"))
    if args.step == "frameslides":
        return frameslides(ids, cfg, dcfg, Path("results/external/lectqa_frameslides.md"))
    if args.step == "v2":
        return v2(ids, cfg, dcfg, Path("results/external/lectqa_v2.md"), cache_only=not args.allow_new_calls,
                  max_cost=args.max_cost)
    return run(ids, cfg, dcfg, args.modes.split(","), Path(args.out), args.repeats,
               metrics=args.metrics, cache_only=args.cache_only)


if __name__ == "__main__":
    raise SystemExit(main())
