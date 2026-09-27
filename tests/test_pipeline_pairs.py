"""link_corpus aligns each (recording, deck) pair on its own."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from conftest import _audio, _txt
from linkrag.core import load_config
from linkrag.link.pipeline import link_corpus

WORDS = ["attention weights here", "optimizer warmup schedule", "clustering of vectors"]


def _encoder(texts_seen):
    vocab = sorted({t for s in texts_seen for t in s.split()})

    def enc(texts):
        m = np.zeros((len(texts), len(vocab)), dtype="float32")
        for r, t in enumerate(texts):
            for w in t.split():
                if w in vocab:
                    m[r, vocab.index(w)] = 1
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    return enc


def test_two_recordings_one_deck_are_aligned_separately() -> None:
    cfg = load_config("configs/default.yaml")
    cfg["link"]["align"]["relatedness_z"] = None          # the gate is tested elsewhere
    rec1 = [replace(_audio(f"r1:a{i}", w, start=10 * i), source_file="r1.mp3") for i, w in enumerate(WORDS)]
    rec2 = [replace(_audio(f"r2:a{i}", w, start=10 * i), source_file="r2.mp3") for i, w in enumerate(reversed(WORDS))]
    deck = [_txt(f"t{i}", i + 1, w) for i, w in enumerate(["attention weights", "optimizer schedule", "vector clustering"])]
    enc = _encoder(WORDS + [u.content for u in deck])
    run = link_corpus(rec1 + rec2, deck, deck, [], encoder=enc, cfg=cfg)
    assert set(run.alignments) == {("r1.mp3", "deck.pdf"), ("r2.mp3", "deck.pdf")}
    assert run.alignment is None and run.gate is None     # several pairs: see alignments / gates
    for (rec, _deck), a in run.alignments.items():
        assert a.similarity.shape == (3, 3)                # one recording against one deck, not the concatenation
    assert {l.src_id.split(":")[0] for l in run.audio_slide} == {"r1", "r2"}

    single = link_corpus(rec1, deck, deck, [], encoder=enc, cfg=cfg)
    assert single.alignment is not None and list(single.alignments) == [("r1.mp3", "deck.pdf")]
    assert [l.dst_id for l in single.audio_slide] == [l.dst_id for l in run.audio_slide if l.src_id.startswith("r1")]
