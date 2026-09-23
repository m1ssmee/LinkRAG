"""LectQA-Vid answer metrics as target T1 defines them.

Source: Shafiq et al., "Intra-Video Temporal-Aware RAG", CMC 88(2) art. 96, 2026, the
evaluation-metrics section, eqs. 29-36 (full-text HTML, read 2026-09-23). They publish no
code, so everything the equations leave open is stated here as our choice.

  token P/R/F1  eqs. 29-30  over token SETS: P = |T∩T̂|/|T̂|, R = |T∩T̂|/|T|
  BLEU          eq. 31      4-gram, uniform weights 1/4, brevity penalty; no smoothing stated
  METEOR        eqs. 32-33  F = 10PR/(R+9P), penalty 0.5·(chunks/matches)^3
  ROUGE-1       eq. 34      unigram RECALL over sets, |T∩T̂|/|T| (not F1)
  similarity    eq. 35      cosine of all-MiniLM-L6-v2 embeddings
  MCQ accuracy  eq. 36      correct / total; Table 5's P/R/F1 are "computed over the
                            predicted and ground-truth option labels"

Our choices where the paper is silent:
* Tokenisation: lowercase, then runs of [a-z0-9]. No stop-word or article removal: eq. 29
  says only "sets of tokens". The same tokens feed every metric.
* BLEU and METEOR: nltk (3.10). nltk's `sentence_bleu` with weights (0.25,)*4 and no
  smoothing is eq. 31 exactly. With no smoothing, an answer with no 4-gram match scores 0,
  which is what eq. 31 gives. `single_meteor_score` defaults (alpha 0.9, beta 3, gamma 0.5)
  are exactly eqs. 32-33; its matching stages are exact, Porter stem and WordNet synonym.
  One known deviation: nltk aligns greedily, so when a word repeats in the reference it can
  count more chunks than the minimal alignment METEOR defines (tested). This affects only
  answers with repeated words, and only through the fragmentation penalty.
  sacrebleu was not used because it has no METEOR.
* MCQ P/R/F1: macro average over the option labels present in gold or prediction; an
  unanswered question counts against recall. The paper does not state the averaging.
* Scores are returned as fractions; the paper reports percentages (×100).
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from statistics import mean
from typing import Sequence

TOKEN = re.compile(r"[a-z0-9]+")
SIM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokens(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


def token_prf(pred: str, ref: str) -> tuple[float, float, float]:
    """Eqs. 29-30, over token sets."""
    p, r = set(tokens(pred)), set(tokens(ref))
    if not p or not r:
        return 0.0, 0.0, 0.0
    inter = len(p & r)
    if not inter:
        return 0.0, 0.0, 0.0
    prec, rec = inter / len(p), inter / len(r)
    return prec, rec, 2 * prec * rec / (prec + rec)


def rouge1(pred: str, ref: str) -> float:
    """Eq. 34: unigram recall over sets."""
    return token_prf(pred, ref)[1]


def bleu4(pred: str, ref: str) -> float:
    """Eq. 31: nltk sentence BLEU, uniform 4-gram weights, no smoothing."""
    from nltk.translate.bleu_score import sentence_bleu
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    with warnings.catch_warnings():            # nltk warns when an n-gram order has no match
        warnings.simplefilter("ignore")
        score = float(sentence_bleu([r], p, weights=(0.25, 0.25, 0.25, 0.25)))
    # nltk substitutes the smallest float for log(0), so a zero precision yields ~1e-232;
    # eq. 31 gives exactly 0 there
    return 0.0 if score < 1e-100 else score


def meteor(pred: str, ref: str) -> float:
    """Eqs. 32-33: nltk METEOR with the paper's parameters (nltk's defaults)."""
    from nltk.translate.meteor_score import single_meteor_score
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    return float(single_meteor_score(r, p, alpha=0.9, beta=3.0, gamma=0.5))


@lru_cache(maxsize=2)
def _sim_model(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SIM_MODEL, device=device)


def similarity(preds: Sequence[str], refs: Sequence[str], device: str = "cpu") -> list[float]:
    """Eq. 35: cosine of all-MiniLM-L6-v2 embeddings, one per (prediction, reference) pair."""
    if not preds:
        return []
    emb = _sim_model(device).encode([*preds, *refs], normalize_embeddings=True)
    n = len(preds)
    return [float(emb[i] @ emb[n + i]) for i in range(n)]


def score_open(preds: Sequence[str], refs: Sequence[str], device: str = "cpu") -> dict[str, float]:
    """Means over the pairs, as fractions: f1, precision, recall, bleu, meteor, rouge1, sim."""
    prf = [token_prf(p, r) for p, r in zip(preds, refs)]
    return {"n": len(preds),
            "precision": mean(x[0] for x in prf), "recall": mean(x[1] for x in prf), "f1": mean(x[2] for x in prf),
            "bleu": mean(bleu4(p, r) for p, r in zip(preds, refs)),
            "meteor": mean(meteor(p, r) for p, r in zip(preds, refs)),
            "rouge1": mean(rouge1(p, r) for p, r in zip(preds, refs)),
            "sim": mean(similarity(preds, refs, device))} if preds else {"n": 0}


def score_mcq(pred: Sequence[str | None], gold: Sequence[str]) -> dict[str, float]:
    """Eq. 36 accuracy, plus macro P/R/F1 over option labels (None = unanswered)."""
    n = len(gold)
    if not n:
        return {"n": 0}
    labels = sorted({g for g in gold} | {p for p in pred if p is not None})
    ps, rs, fs = [], [], []
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in zip(pred, gold))
        fp = sum(p == lab and g != lab for p, g in zip(pred, gold))
        fn = sum(g == lab and p != lab for p, g in zip(pred, gold))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        ps.append(pr)
        rs.append(rc)
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"n": n, "accuracy": sum(p == g for p, g in zip(pred, gold)) / n,
            "precision": mean(ps), "recall": mean(rs), "f1": mean(fs)}
