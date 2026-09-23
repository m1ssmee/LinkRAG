# LectQA-Vid — faithfulness per mode

The 900 repeat-0 answers of the open-ended subset (`lectqa_open_modes.md`: 100 questions per difficulty, seed `lectqa-open-subset-20260923`, 93 videos) · answerer `gpt-5.4-mini-2026-03-17` · judge `gpt-4.1-mini-2025-04-14` temperature 0.0, 1 run(s) per check · `linkrag.generate.verify.verify_answer`: each claim against the text of the units it cites; the first entailing unit wins, with a quotable span required.

**supported**: a cited unit entails the claim. **weak**: the claim cites no unit of the evidence. **unsupported**: no cited unit entails it. **Hallucination rate** = unsupported / claims, pooled over answers (`hallucination_rate`). Supported means the claim is stated in the cited evidence. It does not mean the answer is correct.

| level | mode | answers | claims | supported | weak | unsupported | hallucination rate | answers with no claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| simple | baseline | 100 | 195 | 188 | 0 | 7 | 3.6 % | 12 |
| simple | iterative | 100 | 239 | 216 | 0 | 23 | 9.6 % | 1 |
| simple | full_context | 100 | 228 | 206 | 0 | 22 | 9.6 % | 1 |
| hard | baseline | 100 | 205 | 191 | 0 | 14 | 6.8 % | 24 |
| hard | iterative | 100 | 257 | 249 | 0 | 8 | 3.1 % | 6 |
| hard | full_context | 100 | 257 | 243 | 0 | 14 | 5.4 % | 6 |
| very hard | baseline | 100 | 247 | 238 | 0 | 9 | 3.6 % | 15 |
| very hard | iterative | 100 | 300 | 288 | 0 | 12 | 4.0 % | 7 |
| very hard | full_context | 100 | 301 | 293 | 0 | 8 | 2.7 % | 6 |
| overall | baseline | 300 | 647 | 617 | 0 | 30 | 4.6 % | 51 |
| overall | iterative | 300 | 796 | 753 | 0 | 43 | 5.4 % | 14 |
| overall | full_context | 300 | 786 | 742 | 0 | 44 | 5.6 % | 13 |


LLM cost (this run):

- `gpt-4.1-mini-2025-04-14`: 2260 calls (0 cached) · 667,683 in / 150,006 out · $0.5071
- run total: $0.5071
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 33,483 calls · 28,568,878 in / 1,396,762 out · $22.05 · 6 row(s) unpriced · 14,412,524 tokens estimated
