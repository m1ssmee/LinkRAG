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

from linkrag.core import EvidenceUnit, Link, Location, load_config, setup_logging
from linkrag.costs import cached_completer, record_run
from linkrag.generate.answer import http_completer
from linkrag.index import Index, build_index, default_encoder
from linkrag.link.graph import build_graph
from linkrag.manifest import MANIFEST_NAME, build_manifest, write_manifest
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit, retrieve_linkrag
from linkrag.retrieve.rerank import rerank

RAW = Path("data/raw/lectqa_vid")
PROCESSED = Path("data/processed/lectqa_vid")
LEVELS = ("simple", "hard", "very hard")
THEIRS = {  # Table 4 (open-ended, proposed) and Table 5 (MCQ, proposed), per level
    "open": {"simple": (31.15, 39.80, 0.75), "hard": (19.35, 24.60, 0.68),
             "very hard": (10.24, 15.32, 0.51), "overall": (23.52, 29.76, 0.71)},
    "mcq": {"simple": 68.40, "hard": 56.30, "very hard": 44.20, "overall": 56.30},
}


# ----------------------------------------------------------------- data

def video_links() -> dict[str, str]:
    import docx
    doc = docx.Document(RAW / "video_links.docx")
    out = {}
    for row in doc.tables[0].rows[1:]:
        vid, url = (c.text.strip() for c in row.cells[:2])
        out[vid.replace(" ", "_")] = url
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
                break
        else:
            print(f"fetched {vid}")
    if failures:
        (RAW / "fetch_failures.txt").write_text("\n".join(f"{v}\t{e}" for v, e in failures) + "\n")
        print(f"{len(failures)} video(s) unavailable -> {RAW / 'fetch_failures.txt'}", file=sys.stderr)


# ----------------------------------------------------------------- prepare

def extract_frames(video: Path, out_dir: Path, every_s: float, min_change: float = 0.08) -> list[tuple[float, Path]]:
    """One frame every `every_s` seconds, kept only if it differs from the last kept
    frame (mean absolute pixel difference on a 64x36 grey thumbnail > min_change)."""
    import av
    from PIL import Image
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
        audio = ingest_audio(audio_path, model_size=cfg["models"]["whisper"], device=cfg["device"],
                             compute_type=cfg["models"]["whisper_compute_type"],
                             window_seconds=dcfg["audio_segment_seconds"], segmentation="sentence")
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


def rouge1(pred: str, ref: str) -> float:
    return token_f1(pred, ref)  # unigram ROUGE-1 F is the same computation without stop-word/article removal


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


def run(ids: list[str], cfg: dict, dcfg: dict, modes: list[str], out: Path, repeats: int) -> int:
    qa = load_qa()
    llm = cfg["models"]["llm"]
    complete = cached_completer(http_completer(llm), PROCESSED / "llm_cache" / str(llm.get("model")))
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
                        reply = complete(MCQ_SYSTEM, f"Evidence:\n{ev}\n\nQuestion: {q['question']}\n{opts}\n\nLetter:")
                        choice = mcq_choice(reply, q["options"])
                        gold = q["options"].index(q["answer"]) if q["answer"] in q["options"] else None
                        rows.append({"vid": vid, "kind": "mcq", "level": q["level"], "mode": mode, "rep": rep,
                                     "correct": float(choice is not None and choice == gold), "reply": reply[:40]})
                    else:
                        reply = complete(OPEN_SYSTEM, f"Evidence:\n{ev}\n\nQuestion: {q['question']}\n\nAnswer:")
                        vec = np.asarray(encoder([reply, q["answer"]]), dtype=np.float32)
                        sim = float(vec[0] @ vec[1] / (np.linalg.norm(vec[0]) * np.linalg.norm(vec[1]) + 1e-9))
                        rows.append({"vid": vid, "kind": "open", "level": q["level"], "mode": mode, "rep": rep,
                                     "f1": token_f1(reply, q["answer"]), "rouge1": rouge1_raw(reply, q["answer"]),
                                     "sim": sim, "reply": reply[:200]})
        done = len({r["vid"] for r in rows})
        print(f"{vid}: {len(qa.get(vid, []))} QA · {done} videos done · {time.perf_counter() - t_start:.0f}s")

    if not rows:
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def agg(kind, mode, level, key):
        xs = [r[key] for r in rows if r["kind"] == kind and r["mode"] == mode and (level == "overall" or r["level"] == level)]
        return (statistics.mean(xs), len(xs)) if xs else (float("nan"), 0)

    n_videos = len({r["vid"] for r in rows})
    L = [f"# LectQA-Vid — target T1 — first run", "",
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
            tf, tr, ts = THEIRS["open"][level]
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "prepare", "run"])
    ap.add_argument("--videos", type=int, default=None, help="first N video ids")
    ap.add_argument("--only", default=None, help="comma-separated video ids")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modes", default="baseline,linkrag")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="reports/lectqa_vid_first_run.md")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    dcfg = cfg.get("datasets", {}).get("lectqa_vid", {"audio_segment_seconds": 15, "frame_interval_s": 5, "top_k": 4})
    ids = video_ids(args.videos, args.only)
    if args.step == "fetch":
        fetch(ids)
        return 0
    if args.step == "prepare":
        prepare(ids, cfg, dcfg)
        return 0
    return run(ids, cfg, dcfg, args.modes.split(","), Path(args.out), args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
