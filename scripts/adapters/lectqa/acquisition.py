"""Fetch the videos (yt-dlp, YouTube links from the Mendeley record) and report coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path



from lectqa.common import PROCESSED, RAW, load_qa, video_ids, video_links


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
