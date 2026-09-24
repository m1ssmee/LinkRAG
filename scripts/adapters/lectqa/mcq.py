"""MCQ with options shuffled under a recorded seed."""

from __future__ import annotations

import collections
import json
from pathlib import Path


from linkrag.costs import cached_completer, price_for, record_run, usage_cost
from linkrag.generate.answer import http_completer
from linkrag.index import Index, default_encoder
from linkrag.retrieve.baseline import retrieve_scored
from linkrag.retrieve.linkrag import RetrievedUnit

from lectqa.common import LEVELS, MCQ_SYSTEM, PROCESSED, THEIRS, evidence_block, load_qa, mcq_choice


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
