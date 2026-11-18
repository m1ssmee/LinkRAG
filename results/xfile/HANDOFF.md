# XFile: handoff (2026-11-17, updated 2026-11-18)

> **m1ssmee, 2026-11-18 (latest).** Rebuilt on m1ssmee's machine. `prepare` is not byte-reproducible
> (PyMuPDF writes a fresh document ID and timestamp into each cut notes PDF), so a fresh `prepare` gives
> manifest `26ddfe58a5a8f021`. With Rakshit's four cut notes PDFs (checked against `SHA256SUMS.txt`) and
> **no** re-run of `prepare`, ingest gives `a086ca1c22dc1d93`; `make test` passed.
> **Done since:** apply-review (89 accept, 1 edit, 8 reject -> 90 questions); shortcut 6 (all 360
> whole-source answers re-made one question per subagent); shortcut 4 (gold check at 3 runs on the
> reviewed set): **86 kept, 4 dropped, 6 relabelled; 53 cross-file (44 split + 9 deictic)**, below the
> PLAN target of 60. Test half: 21 cross-file + 17 single-source; tune 23 + 13; lightning 9 + 3.
> Result: `results/xfile/gold_check_3runs.md`/`.json`, gold `tests/regression/xfile_questions.jsonl`.
> Before trusting it, a transcript audit found 133 Haiku answers made without reading the whole
> context; they were redone (DESIGN.md (o), `scripts/agent_read_audit.py`). **Run that audit on every
> answer round before `import`.** Old shared-context answers: `data/processed/xfile/haiku_answers_shared_ctx_backup/`.
> Next: shortcut 5 (human audit sheet), then Step 4.
>
> **Steps 4 and 5 done (m1ssmee, 2026-11-18).** Frozen default knobs, no tuning. Tune half: check only
> (`eval_tune.md`). Test half (gp_scaling, mimic_data, rl_mdp; 38 questions; cache-only replay;
> `eval_test.md`): **no measurable gain from links**. Headline linkrag + complementarity vs baseline +
> complementarity, cross-file 38.1 vs 47.6 %, -9.5 [-16.7, +0.0]; all -3.5 [-9.1, +5.1]; linkrag +
> complementarity equals plain top-8 exactly. Report: `results/xfile/xfile_benchmark.md`; README
> contribution 5 and Limitation updated. `reply_cache.tar.gz` refreshed (every reply behind both halves).
> **Not done:** lightning row (deadline); m1ssmee's 26 audit rows; `cut_pages` reproducibility (then a
> re-ingest and re-stamp as a separate correction); the 3 informational redundancy directions. To replay
> any table: rebuild as above, `tar -xzf results/xfile/reply_cache.tar.gz`, and run the command in the
> report header with `--answerer-cache-only --judge-cache-only`.


From Rakshit's session to m1ssmee. The plan is [PLAN.md](PLAN.md), and the policy change is
recorded in `DESIGN.md` (*Dataset policy*, amended 2026-11-17, and priority (vi)).

To save usage, this first pass took the **shortcuts listed under "Undo these shortcuts"**.
Redo them before reporting any number.

## Where things stand

| Step | State |
|---|---|
| 0. Policy note | Done: `DESIGN.md`, plus known issue (m). |
| 1. Corpus | Done. 7 topics × (MaViLS transcript, MaViLS deck, separate notes PDF) = 21 files, 2,210 units, manifest **`a086ca1c22dc1d93`**. `med_imgproc` was dropped (image-only deck, issue (m)). |
| 1b. Links | Done: `results/xfile/links.jsonl` (2,222 links, stamped `a086ca1c22dc1d93`; unit ids and scores only). Copy it into place; don't rebuild (~4 h CPU). **The audio↔slide gate leaks cross-lecture links**: 9/49 pairs kept, 5 true + 4 false, and mimic_data and visual_system failed against their own decks. See DESIGN.md known issue (n). |
| 2. Judge | Done at **3 runs**: unit κ **0.81** vs gpt-4.1-mini (bar 0.70), 0.93 vs gpt-5.4; grades 98/100; types 24/25. See `results/judge_agreement_offline_claude_agents_judge_3runs.md`. The 1-run report is kept beside it. |
| 2b. Redundancy gate | Done (deck→transcript, the gate pair): 6 KEEP, **lightning REJECT** (census 67.3 %); psych_methods KEEP by census (57.1 %). Decision log in `scripts/dataset/README.md`. The 3 informational directions are only partly answered. |
| 3. Questions | 98 drafted by Opus subagents (`tests/regression/xfile_drafts.jsonl`). `verify_gold` kept 97: **64 verified cross-file** (54 split + 10 deictic), 33 single-source controls, 1 dropped (LN12). **1 run.** See `gold_check_1run.md` / `.json`. |
| 3b. Human review | Sheet ready: `reports/xfile_review.csv`. It holds a per-question `kind` (bridge / conjunction / deictic / control), Claude's recommendation (8 reject, 6 check) and the gold-check result. **Rakshit's half (RL, GP, NF, MD) is decided:** 50 accept, 1 edit (GP10), 5 reject. RL01–RL10 were looked at row by row first (RL09/RL10 against the slide images); the rest first took Claude's recommendation. m1ssmee's half (PS, VS, LN) is decided: 39 accept, 3 reject (PS07, VS01, LN02), first by applying Claude's recommendation (LN08 'check' resolved as accept). **Update 2026-11-18:** Rakshit then read all 98 rows individually against their evidence; no decision changed (`reviewer_note` records both steps). |
| 4. Evaluation | **Done 2026-11-18** (`eval_test.md`, `eval_tune.md`). Code ready: `compare_retrieval.py` goes through the reply cache, `--correctness` grades answers (by type and kind), `--topics` selects tune/test halves. Not run yet. |
| 5. Report | **Done 2026-11-18** (`xfile_benchmark.md`, README). **Paper:** `paper/paper.html` is restructured around LinkRAG + XFile; §4.1 waits for results. After the test-half run, `python paper/xfile_fill.py <test-half out>.questions.json && python paper/figures.py` fills Table 3 and Figure 5 (same topic bootstrap as `xfile.py paired`). The `[[XFILE]]` sentences are written by hand. |

## How the LLM calls are made (no API keys)

`provider: agent_queue` backends (`claude_agents` = Haiku answerer, `claude_agents_judge` =
Sonnet judge, in `configs/*.yaml`) answer through the reply cache. See
`src/linkrag/agent_queue.py` and `scripts/agent_queue.py`. One round:

```bash
export LINKRAG_LLM_BACKEND=claude_agents LINKRAG_JUDGE_BACKEND=claude_agents_judge
export LINKRAG_AGENT_QUEUE=data/processed/agent_queue/<run>.jsonl
<script> ...                                   # collect: misses are queued, output is NOT a result
python scripts/agent_queue.py status
python scripts/agent_queue.py batches --out data/processed/agent_queue/<run>_r1
#  -> one subagent per batch folder (prompt templates below), haiku for claude-haiku-5-5
#     batches, sonnet for claude-sonnet-5-5 batches
python scripts/agent_queue.py import --batches data/processed/agent_queue/<run>_r1
<script> ...                                   # repeat until `status` shows 0 pending
<script> ... --answerer-cache-only --judge-cache-only   # final, clean replay = the result
```

Stages that depend on earlier replies (answer, then grade) take one round each. A `c*` batch
holds one shared long context: its user message is `context.txt` + `N.question.txt`.

Subagent prompt for a `b*` batch:

> You are acting as the LLM behind an evaluation harness. Folder: `<abs path>`. For each N in
> manifest.json there is N.system.txt and N.user.txt. For EVERY N write to replies/N.txt exactly
> the raw reply the model should return, following the system prompt and its output format to the
> letter (JSON only where asked, no code fences, no commentary). Treat each prompt independently;
> use only the text in the prompt; do your own reasoning, no scripts; quoted spans copied exactly;
> don't modify other files.

For a `c*` batch, also say: *read system.txt, then ALL of context.txt in offset/limit chunks;
the user message of item N is context.txt followed by N.question.txt; answer ONLY from the
context, else exactly NOT ANSWERABLE.*

## Rebuild on a new machine

```bash
make setup && brew install tesseract
git clone https://github.com/andererka/MaViLS data/raw/mavils && git -C data/raw/mavils checkout b3c55a869a24a8f3f3881aa3e89fea9b1c81c0ad
# download configs/xfile_topics.yaml `sources` into data/raw/xfile/_sources/, check each sha256
python scripts/adapters/xfile.py prepare
python scripts/ingest.py $(python scripts/adapters/xfile.py files) --config configs/xfile.yaml --mode linkrag --max-cost 0
# expect "corpus manifest a086ca1c22dc1d93". If it differs, the cached replies below won't hit.
tar -xzf results/xfile/reply_cache.tar.gz       # restores verify_cache + judge_cache_offline (+ the superseded shared-context answers)
cp results/xfile/links.jsonl data/processed/xfile/links.jsonl   # don't rebuild (~4 h CPU)
```

## Undo these shortcuts (in this order)

1. **Judge agreement at 3 runs.**
   `LINKRAG_JUDGE_BACKEND=claude_agents_judge python scripts/eval/judge_agreement_offline.py --runs 3`.
   Run 0 is cached; runs 1 and 2 are new prompts.
2. **Redundancy intake gate (Step 2).** For each topic, `scripts/dataset/redundancy.py --index
   data/processed/xfile/index ...` with the Sonnet judge. Record KEEP/REJECT in
   `scripts/dataset/README.md`. It was skipped; it decides whether a topic's deck mostly repeats
   its talk.
3. **Human review.** Fill `reports/xfile_review.csv` (decision column; the sheet splits topics
   between you two). Check RL09/RL10 against the slide images, since their answers come from
   OCR. Then run `python scripts/adapters/xfile.py apply-review`, which writes
   `tests/regression/xfile_proposed.jsonl`.
4. ✅ *(done 2026-11-18, `gold_check_3runs.md`)* **Gold check at 3 runs, on the reviewed set**: `scripts/eval/verify_gold.py --config
   configs/xfile.yaml --corpus xfile --proposed tests/regression/xfile_proposed.jsonl --out
   tests/regression/xfile_questions.jsonl --runs 3 --entailment llm --max-cost 0`. Run-0 answers
   and verdicts are cached. Edited questions are new prompts.
5. 🔶 *(sheet ready 2026-11-18, to be filled by hand)* `reports/xfile_audit_sheet.csv`: 50 unit verdicts
   (10 %, min 5 per type, plus at least 5 rows from each of GP, NF, MD, PS, VS, LN, the topics whose review
   took Claude's recommendation unread). Each person audits the **other's** topics (`auditor` column):
   m1ssmee RL/GP/NF/MD (26 rows), Rakshit PS/VS/LN (24 rows). Fill `agree` with y/n (and `auditor_note`
   on n), never by Claude, then `python scripts/eval/audit_sample.py --out reports/xfile_audit_sheet.csv --score`.
   Built with `--verdicts results/xfile/gold_check_3runs.json --min-per-prefix GP,NF,MD,PS,VS,LN:5
   --assign "m1ssmee:RL,GP,NF,MD;Rakshit:PS,VS,LN"`. Original item: **Human audit**: `scripts/eval/audit_sample.py` (10%, at least 5 per type). Fill the sheet
   (DESIGN.md known issue (d)).
6. ✅ *(done 2026-11-18, before 4)* *Optional:* re-answer the whole-source runs with `--share-max-items 1`, so each question is
   answered in a context of its own. In the shared-context batches a subagent saw ~14 questions
   on the same context.

## Finish the redundancy gate

Re-running collection rebuilds the same queue, because the prompts are deterministic:

```bash
export LINKRAG_JUDGE_BACKEND=claude_agents_judge LINKRAG_AGENT_QUEUE=data/processed/agent_queue/redundancy.jsonl
for t in rl_mdp gp_scaling nfa_closure mimic_data psych_methods visual_system lightning; do
  python scripts/dataset/redundancy.py --config configs/xfile.yaml --index data/processed/xfile/index \
    --corpus xfile_$t --topic $t --sample 50 --runs 3 --entailment llm --max-cost 0 --reports-dir reports
done      # first pass queues; after import, the same loop writes reports/redundancy_xfile_<topic>.md
```

Tell the subagents to **read every passage in full**. Several first-wave subagents cut passages to
their first ~1,000 characters, and those batches were cleared and must be redone.

## Step 4: running the evaluation

- Split by topic into tune and test halves **before any run**, and write the split into PLAN.md.
  Knobs are set on tune only (DESIGN.md *Reporting convention*).
- Headline comparison: `linkrag + complementarity` vs the no-links control `baseline +
  complementarity`. The control was fixed this session: `rerank_graph()` in compare_retrieval.py
  no longer hands the graph to non-link modes. Run 3 repeats, with a paired bootstrap over
  **topics**.

## Caveats to carry into any report

- Subagent runs can be re-scored but not re-generated. Haiku is both the answerer and the drafter's
  model family. Label results accordingly.
- Drafts came from Claude, and the drafter checked fact exclusivity with grep. That is why
  64/70 cross-file drafts survived, against 4/16 on pilot01. A human review is what makes them gold.
- Answer replies sometimes give "the part the context supports + NOT ANSWERABLE for the rest".
  The grader correctly fails those. One batch (b025) used a bare NOT ANSWERABLE instead; the
  outcome is the same.
