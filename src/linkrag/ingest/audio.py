"""Audio -> EvidenceUnits via faster-whisper (CTranslate2, int8 on CPU).

baseline: fixed ~30s transcript windows, indexed like any other text. This is
          P2's audio side without even its intra-video timestamp fusion.
linkrag:  identical units; word timestamps are kept in metadata so the linker
          can align a deictic phrase to the slide that was on screen then.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from linkrag.core import EvidenceUnit, Location, stage_timer


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    """One whisper decoding segment -- roughly a sentence or clause, ending on
    punctuation far more often than an arbitrary time offset does."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


# Whisper conditions on at most ~224 prompt tokens; past that the tail is dropped.
ASR_PROMPT_MAX_TERMS = 75

_TERM = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def build_asr_prompt(slide_texts: list[str], max_terms: int = ASR_PROMPT_MAX_TERMS) -> str:
    """Domain vocabulary for whisper's `initial_prompt`, mined from the slide deck.

    Whisper conditions the decoder on this text, biasing it toward spellings it
    would otherwise smooth into common words. On pilot01 the untuned model wrote
    *interest* for **ingest** throughout -- the single most load-bearing term in
    the talk, and one BM25 then cannot match.

    Ranking is by "distinctive to this deck": a term scores on how many slides it
    appears on, but only if it is not an everyday English word. That keeps the
    prompt on jargon and proper nouns, which is what whisper actually needs help
    with.
    """
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "can", "you", "our",
        "all", "not", "but", "how", "why", "who", "when", "where", "what", "which", "was",
        "have", "has", "his", "her", "its", "one", "two", "use", "using", "used", "each",
        "more", "most", "than", "then", "they", "them", "there", "these", "those", "will",
        "would", "could", "should", "may", "might", "must", "into", "over", "under", "also",
        "such", "some", "any", "per", "via", "based", "different", "same", "other", "only",
        "time", "cost", "low", "high", "new", "old", "first", "second", "third", "page",
    }
    slides_containing: dict[str, set[int]] = {}
    casing: dict[str, str] = {}
    for i, text in enumerate(slide_texts):
        for match in _TERM.findall(text):
            key = match.lower()
            if key in stop:
                continue
            slides_containing.setdefault(key, set()).add(i)
            # keep the most "shouty" casing seen: NoScope beats noscope
            if key not in casing or sum(c.isupper() for c in match) > sum(
                c.isupper() for c in casing[key]
            ):
                casing[key] = match
    def jargon(term: str) -> int:
        """Signals a word whisper is likely to smooth away: digits (YOLOv2),
        internal capitals (NoScope, ResNet), or a hyphen (Top-K)."""
        return int(
            any(c.isdigit() for c in term)
            or any(c.isupper() for c in term[1:])
            or "-" in term
        )

    # Slide frequency first, then jargon -- an alphabetical tiebreak silently
    # dropped YOLOv2 and kept 'Car'.
    ranked = sorted(
        slides_containing,
        key=lambda k: (-len(slides_containing[k]), -jargon(casing[k]), -len(k), k),
    )[:max_terms]
    if not ranked:
        return ""
    # Note: hyphens are kept. De-hyphenating was tried on pilot01 to stop
    # whisper emitting "type -k" for spoken "top-K"; a full re-run produced a
    # byte-for-byte equivalent transcript, so the hyphen was not the cause and
    # the workaround was removed. Do not re-try it without new evidence.
    terms = ", ".join(casing[k] for k in ranked)
    return f"A technical conference talk. Terminology used: {terms}."


@lru_cache(maxsize=2)
def _load_model(size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel  # heavy import, deferred to first use

    return WhisperModel(size, device=device, compute_type=compute_type)


def segment_words(words: list[Word], window_seconds: float = 30.0) -> list[list[Word]]:
    """Group consecutive words into ~window_seconds buckets.

    A bucket closes on the first word that would push its span past the window,
    so units end on a word boundary rather than mid-syllable at a hard 30.0s cut.
    """
    if not words:
        return []
    buckets: list[list[Word]] = [[]]
    anchor = words[0].start
    for word in words:
        if buckets[-1] and word.end - anchor > window_seconds:
            buckets.append([])
            anchor = word.start
        buckets[-1].append(word)
    return [b for b in buckets if b]


TERMINAL = re.compile(r"[.!?][\"')\]]?$")


def split_sentences(words: list[Word]) -> list[list[Word]]:
    """Cut the word stream after every word carrying terminal punctuation.

    Whisper attaches punctuation to words, so this finds real sentence ends.
    Note it does NOT use whisper's own `segments`: those are decoder chunks, and
    on the pilot lecture only 26% of them ended on a sentence -- packing them
    left 74% of unit boundaries mid-sentence, marginally worse than fixed windows.
    """
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if TERMINAL.search(word.text):
            sentences.append(current)
            current = []
    if current:  # trailing words with no final punctuation
        sentences.append(current)
    return sentences


def pack(groups: list[list[Word]], window_seconds: float = 30.0) -> list[list[Word]]:
    """Greedily merge whole groups into ~window_seconds buckets, never splitting
    one. A group longer than the window becomes its own bucket."""
    buckets: list[list[Word]] = []
    for group in groups:
        if not group:
            continue
        if buckets and group[-1].end - buckets[-1][0].start <= window_seconds:
            buckets[-1].extend(group)
        else:
            buckets.append(list(group))
    return buckets


def transcribe_segments(
    path: str | Path,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    initial_prompt: str | None = None,
) -> list[Segment]:
    model = _load_model(model_size, device, compute_type)
    segments, _info = model.transcribe(
        str(path), word_timestamps=True, initial_prompt=initial_prompt or None
    )
    return [
        Segment(
            s.start,
            s.end,
            s.text.strip(),
            tuple(Word(w.start, w.end, w.word.strip()) for w in (s.words or [])),
        )
        for s in segments
    ]




def load_frozen_transcript(path: str | Path) -> list[Word]:
    """Words from a persisted transcript, so a frozen corpus is never re-decoded.

    Transcription is the single most expensive and least deterministic step in the
    pipeline. Once a corpus is frozen, every later phase must read the same words
    or cross-phase comparisons quietly stop meaning anything.
    """
    payload = json.loads(Path(path).read_text())
    return [Word(float(a), float(b), str(t)) for a, b, t in payload["words"]]


def frozen_transcript_for(audio: str | Path, directory: str | Path | None) -> Path | None:
    """`<directory>/<stem>.frozen.json` if it exists, else None."""
    if not directory:
        return None
    candidate = Path(directory) / f"{Path(audio).stem}.frozen.json"
    return candidate if candidate.exists() else None


def ingest_audio(
    path: str | Path,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    window_seconds: float = 30.0,
    segmentation: str = "sentence",
    initial_prompt: str | None = None,
    transcript: str | Path | None = None,
    backend: str = "local",
    cfg: dict | None = None,
    freeze_to: str | Path | None = None,
) -> list[EvidenceUnit]:
    """segmentation="sentence" merges whole whisper segments up to the window.
    "fixed" is the naive equal-time windowing the baseline papers use -- kept so
    the choice stays an ablation rather than a silent assumption.

    `backend`: "local" = faster-whisper here; "openai" = whisper-1 with word
    timestamps (`ingest.asr_openai`), chunked at silence points when the file is over
    the upload limit. Both produce the same words, so everything downstream of this
    function is identical. `freeze_to` writes the transcript so the next run reuses
    it -- the same freeze the local path relies on."""
    if segmentation not in ("sentence", "fixed"):
        # Validate before transcribing: whisper on a lecture is minutes of work
        # to then throw away on a typo.
        raise ValueError(f"unknown segmentation {segmentation!r}: use 'sentence' or 'fixed'")

    path = Path(path)
    if backend not in ("local", "openai"):
        raise ValueError(f"unknown asr_backend {backend!r}: use 'local' or 'openai'")
    ingest_audio.last_transcript = None                # type: ignore[attr-defined]
    with stage_timer(
        "ingest.audio", file=path.name, seg=segmentation,
        src="frozen" if transcript else backend,
    ) as t:
        if transcript is not None:
            words = load_frozen_transcript(transcript)
            ingest_audio.last_transcript = str(transcript)   # type: ignore[attr-defined]
        elif backend == "openai":
            from linkrag.ingest.asr_openai import transcribe, write_transcript
            words, meta = transcribe(path, cfg or {}, prompt=initial_prompt)
            t["asr_backend"] = "openai"
            t["audio_seconds"] = meta["audio_seconds"]
            if freeze_to:
                write_transcript(words, freeze_to, path, meta)
                ingest_audio.last_transcript = str(freeze_to)   # type: ignore[attr-defined]
        else:
            segments = transcribe_segments(
                path, model_size=model_size, device=device, compute_type=compute_type,
                initial_prompt=initial_prompt,
            )
            words = [w for seg in segments for w in seg.words]
        if segmentation == "sentence":
            buckets = pack(split_sentences(words), window_seconds)
        else:  # "fixed" -- validated at entry
            buckets = segment_words(words, window_seconds)
        texts = [" ".join(w.text for w in b) for b in buckets]
        spans = [(b[0].start, b[-1].end) for b in buckets]

        units = [
            EvidenceUnit(
                id=f"{path.stem}:a{i}",
                modality="audio",
                content=text,
                source_file=str(path),
                location=Location(start_s=start, end_s=end),
                # Word times are what the linker needs to place a deictic phrase.
                metadata={"words": [(w.start, w.end, w.text) for w in bucket]},
            )
            for i, (text, (start, end), bucket) in enumerate(zip(texts, spans, buckets))
        ]
        t["words"] = len(words)
        t["units"] = len(units)
    return units
