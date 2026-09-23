#!/usr/bin/env python3
"""Propose a question set for a candidate corpus by modality diff.

The method the pilot01 set was written by hand with (`reports/question_proposal_pilot01.md`),
automated: score every term by how *exclusive* it is to one source (transcript / deck /
paper), then ask the model to write questions around units that carry exclusive terms.

  single-source  one unit whose exclusive terms are the answer's load-bearing words
  cross-modal    two units from different sources that are topically close (cosine)
                 but each carry their own exclusive terms, so neither alone answers it

Output is `tests/regression/<corpus>_proposed.jsonl` in the format
`scripts/eval/verify_gold.py` consumes. **Nothing here decides a question's type** --
the proposal only says what it was built to be; verification decides what it is.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from linkrag.core import EvidenceUnit, load_config, refuse_strong_in_batch, set_max_cost, setup_logging
from linkrag.costs import cached_completer, price_for, record_run
from linkrag.eval.redundancy import role_of
from linkrag.eval.verify_gold import parse_json
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder, tokenize
from linkrag.manifest import MANIFEST_NAME, load_manifest

SYSTEM = ("You write questions for a retrieval benchmark over a recorded lecture. You only "
          "output JSON. A good question is answerable from the evidence you are given, names "
          "specific things, and could not be answered from general knowledge.")

SINGLE_PROMPT = """Evidence from the {role} of a lecture:
\"\"\"{text}\"\"\"

Terms that appear ONLY in this source across the whole corpus: {terms}

Write one question whose answer is in this evidence and uses at least one of those terms.

Output JSON only: {{"question": "...", "expected_answer": "<only the facts the question
asks for, nothing more>"}}"""

CROSS_PROMPT = """Two pieces of evidence from the same lecture, in different sources.

A ({role_a}):
\"\"\"{text_a}\"\"\"
Terms only in the {role_a}: {terms_a}

B ({role_b}):
\"\"\"{text_b}\"\"\"
Terms only in the {role_b}: {terms_b}

Write ONE question that needs a fact from A **and** a fact from B -- neither piece alone
may answer it. Use at least one exclusive term from each side.

Output JSON only: {{"question": "...", "expected_answer": "<only the facts the question
asks for, nothing more>"}}"""

ROLE_LABEL = {"transcript": "spoken transcript", "deck": "slide deck", "paper": "paper"}
SINGLE_TYPE = {"transcript": "audio_only", "deck": "slides_only", "paper": "paper_only"}
STOP = set("the and for with that this from are can you our all not but how why who when where what "
           "which was have has his her its one two use using used each per into over more most some "
           "such then than they them there these those very will would could should about after also".split())


def exclusive_terms(units: list[EvidenceUnit]) -> dict[str, Counter]:
    """role -> Counter of terms that appear in that role and in no other."""
    by_role: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        for t in tokenize(u.content):
            if len(t) > 2 and t not in STOP and not t.isdigit():
                by_role[role_of(u)][t] += 1
    out: dict[str, Counter] = {}
    for role, counts in by_role.items():
        others = set().union(*[set(c) for r, c in by_role.items() if r != role]) if len(by_role) > 1 else set()
        out[role] = Counter({t: n for t, n in counts.items() if t not in others})
    return out


def unit_terms(unit: EvidenceUnit, excl: Counter, k: int = 6) -> list[str]:
    seen = [t for t in dict.fromkeys(tokenize(unit.content)) if t in excl]
    return sorted(seen, key=lambda t: -excl[t])[:k]


def ask(complete, system: str, prompt: str) -> dict | None:
    reply = parse_json(complete(system, prompt))
    q, a = str(reply.get("question", "")).strip(), str(reply.get("expected_answer", "")).strip()
    return {"question": q, "expected_answer": a} if q and a else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--single-per-role", type=int, default=4)
    ap.add_argument("--cross-per-pair", type=int, default=4)
    ap.add_argument("--min-words", type=int, default=25, help="skip units shorter than this")
    ap.add_argument("--max-cost", type=float, required=True,
                    help="USD budget for this run (required; 0 = cache replays and free backends only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    set_max_cost(cfg, args.max_cost)
    refuse_strong_in_batch(cfg)
    index = Index.load(args.index)
    units = [u for u in index.units if len(u.content.split()) >= args.min_words]
    manifest = load_manifest(Path(args.index).parent / MANIFEST_NAME) or {}
    excl = exclusive_terms(units)
    by_role: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for u in units:
        by_role[role_of(u)].append(u)
    print(f"{len(units)} units · roles " + ", ".join(f"{r}={len(v)}" for r, v in by_role.items())
          + " · exclusive terms " + ", ".join(f"{r}={len(c)}" for r, c in excl.items()))

    llm = cfg["models"]["llm"]
    complete = cached_completer(http_completer(llm), Path("data/processed/verify_cache") /
                               manifest.get("hash", args.corpus) / f"{llm.get('model')}-propose",
                               max_cost_usd=args.max_cost,
                               price=price_for(str(llm.get("model")), cfg["models"].get("pricing")))

    encoder = default_encoder(index.embedding_model, cfg["device"], index.normalize)
    encoder([""])
    vec = {u.id: v for u, v in zip(units, np.asarray(encoder([u.content for u in units]), dtype="float32"))}

    rows, qid = [], 0
    # ---- single-source: the units richest in exclusive terms
    for role, group in sorted(by_role.items()):
        ranked = sorted(group, key=lambda u: -sum(excl[role][t] for t in set(tokenize(u.content)) & set(excl[role])))
        for unit in ranked[:args.single_per_role]:
            terms = unit_terms(unit, excl[role])
            if not terms:
                continue
            got = ask(complete, SYSTEM, SINGLE_PROMPT.format(role=ROLE_LABEL[role],
                                                             text=" ".join(unit.content.split())[:2000],
                                                             terms=", ".join(terms)))
            if not got:
                continue
            qid += 1
            rows.append({"qid": f"P{qid:02d}", "type": SINGLE_TYPE[role], **got,
                         "gold_units": [locator_of(unit)], "gold_terms": {"slide": terms[:2], "audio": []},
                         "built_from": {"units": [unit.id], "exclusive_terms": terms}})
            print(f"  {rows[-1]['qid']} {SINGLE_TYPE[role]:<12} {got['question'][:76]}")

    # ---- cross-source: topically close pairs, each side carrying its own exclusive terms
    roles = sorted(by_role)
    for i, ra in enumerate(roles):
        for rb in roles[i + 1:]:
            a_units = [u for u in by_role[ra] if unit_terms(u, excl[ra])]
            b_units = [u for u in by_role[rb] if unit_terms(u, excl[rb])]
            if not a_units or not b_units:
                continue
            A = np.stack([vec[u.id] for u in a_units]); B = np.stack([vec[u.id] for u in b_units])
            A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
            B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
            sim = A @ B.T
            pairs = sorted(((float(sim[x, y]), x, y) for x in range(len(a_units)) for y in range(len(b_units))),
                           reverse=True)
            used_a, used_b, made = set(), set(), 0
            for score, x, y in pairs:
                if made >= args.cross_per_pair:
                    break
                if x in used_a or y in used_b or score < 0.45:
                    continue
                ua, ub = a_units[x], b_units[y]
                ta, tb = unit_terms(ua, excl[ra]), unit_terms(ub, excl[rb])
                got = ask(complete, SYSTEM, CROSS_PROMPT.format(
                    role_a=ROLE_LABEL[ra], text_a=" ".join(ua.content.split())[:1200], terms_a=", ".join(ta),
                    role_b=ROLE_LABEL[rb], text_b=" ".join(ub.content.split())[:1200], terms_b=", ".join(tb)))
                if not got:
                    continue
                used_a.add(x); used_b.add(y); made += 1; qid += 1
                rows.append({"qid": f"P{qid:02d}", "type": "cross_modal_split", **got,
                             "gold_units": [locator_of(ua), locator_of(ub)],
                             "gold_terms": {"slide": tb[:2], "audio": ta[:2]},
                             "built_from": {"units": [ua.id, ub.id], "cosine": round(score, 3),
                                            "exclusive_terms": {ra: ta, rb: tb}}})
                print(f"  {rows[-1]['qid']} cross {ra}/{rb} (cos {score:.2f}) {got['question'][:60]}")

    out = Path(args.out or f"tests/regression/{args.corpus}_proposed.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"_meta": {"purpose": f"Proposed questions for {args.corpus} by modality diff "
                                 f"(scripts/dataset/propose_questions.py). Types are what each question was "
                                 f"BUILT to be; verification decides what it is.",
                      "corpus_manifest": manifest.get("hash"), "model": str(llm.get("model")),
                      "single_per_role": args.single_per_role, "cross_per_pair": args.cross_per_pair}}
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [meta, *rows]) + "\n")
    footer = record_run("scripts/dataset/propose_questions.py", f"{args.corpus} question proposal",
                        [(str(llm.get("model")), complete.usage)], cfg["models"].get("pricing"))
    print("\n".join(footer))
    print(f"\n{len(rows)} questions -> {out}")
    return 0


def locator_of(unit: EvidenceUnit) -> dict:
    loc = {"source": Path(unit.source_file).name}
    if unit.location.page is not None:
        loc["page"] = unit.location.page
    else:
        loc["start_s"] = round(unit.location.start_s or 0.0, 1)
        loc["end_s"] = round(unit.location.end_s or 0.0, 1)
    return loc


if __name__ == "__main__":
    raise SystemExit(main())
