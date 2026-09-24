# LectQA-Vid — faithfulness per mode

The 900 repeat-0 answers of the open-ended subset (`lectqa_open_modes.md`: 100 questions per difficulty, seed `lectqa-open-subset-20260923`, 93 videos) · answerer `gpt-5.4-mini-2026-03-17` · judge `gpt-4.1-mini-2025-04-14` temperature 0.0, 1 run(s) per check · `linkrag.generate.verify.verify_answer`: each claim against the text of the units it cites; the first entailing unit wins, with a quotable span required.

**supported**: a cited unit entails the claim. **weak**: the claim cites no unit of the evidence. **unsupported**: no cited unit entails it. **Hallucination rate** = unsupported / claims, pooled over answers (`hallucination_rate`). Supported means the claim is stated in the cited evidence. It does not mean the answer is correct.

> *Corrected 2026-09-24 (judge cache replay):* iterative, hard: supported 249 → 250, unsupported 8 → 7, rate 3.1 % → 2.7 %; iterative overall 43 → 42 unsupported, 5.4 % → 5.3 %. One claim appears in two answers with an identical judge prompt, and its two stored judge replies differ: one is valid JSON followed by a stray `}`, which the parser rejects and scores as *no*. The old replay handed the two stored replies out in thread order; the cache is now keyed by answer and unit (commit "Judge cache keyed on prompt and answer id"), and a pre-keying entry is read by run index, so both answers now get the first, parseable reply. Numbers above are the corrected ones; the LLM cost below is the original run's (the re-render made no calls).

> *Corrected 2026-09-24 (JSON parser):* 33 claim verdicts moved, all unsupported → supported. `parse_json` sliced from the first `{` to the last `}` and rejected a judge reply that was valid JSON followed by a stray `}`; a rejected reply scores *no*. It now decodes the first object and ignores trailing text. Every cached judge reply was scanned: 39 of 2,260 here parse differently (36 say *yes*), none elsewhere changes a field any consumer reads. Old → new: simple baseline 7/195 = 3.6 % → 3.1 %; simple iterative 23/239 = 9.6 % → 7.1 %; simple full_context 22/228 = 9.6 % → 7.0 %; hard baseline 14/205 = 6.8 % → 5.4 %; hard iterative 7/257 = 2.7 % → 0.8 %; hard full_context 14/257 = 5.4 % → 4.3 %; very hard baseline 9/247 = 3.6 % → 2.8 %; very hard iterative 12/300 = 4.0 % → 2.0 %; very hard full_context 8/301 = 2.7 % → 2.3 %; overall baseline 30/647 = 4.6 % → 3.7 %; overall iterative 42/796 = 5.3 % → 3.1 %; overall full_context 44/786 = 5.6 % → 4.3 %. The numbers above are the corrected ones; the LLM cost below is the original run's.

| level | mode | answers | claims | supported | weak | unsupported | hallucination rate | answers with no claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| simple | baseline | 100 | 195 | 189 | 0 | 6 | 3.1 % | 12 |
| simple | iterative | 100 | 239 | 222 | 0 | 17 | 7.1 % | 1 |
| simple | full_context | 100 | 228 | 212 | 0 | 16 | 7.0 % | 1 |
| hard | baseline | 100 | 205 | 194 | 0 | 11 | 5.4 % | 24 |
| hard | iterative | 100 | 257 | 255 | 0 | 2 | 0.8 % | 6 |
| hard | full_context | 100 | 257 | 246 | 0 | 11 | 4.3 % | 6 |
| very hard | baseline | 100 | 247 | 240 | 0 | 7 | 2.8 % | 15 |
| very hard | iterative | 100 | 300 | 294 | 0 | 6 | 2.0 % | 7 |
| very hard | full_context | 100 | 301 | 294 | 0 | 7 | 2.3 % | 6 |
| overall | baseline | 300 | 647 | 623 | 0 | 24 | 3.7 % | 51 |
| overall | iterative | 300 | 796 | 771 | 0 | 25 | 3.1 % | 14 |
| overall | full_context | 300 | 786 | 752 | 0 | 34 | 4.3 % | 13 |

<details><summary>The 33 moved claims (level, mode, question index, claim)</summary>

| level | mode | question | claim |
|---|---|---:|---|
| hard | baseline | 426 | The video visualizes the inside of an LLM as a glowing web of connections. |
| hard | baseline | 726 | The narrator says, “You've just dipped your toes into the world of prompt engineering.” |
| hard | baseline | 800 | The narrator says Excel is excellent for explaining findings to non-technical colleagues. |
| hard | full_context | 53 | The narrator says the qualities of vehicles also apply to programming languages. |
| hard | full_context | 445 | LLMs work purely based on the data they have been trained on. |
| hard | full_context | 564 | The narrator says NLP can process and analyze vast amounts of text data. |
| hard | iterative | 85 | Python replaces semi-colons with indentation. |
| hard | iterative | 564 | At 29.5s–42.1s, the narrator says NLP enables computers to process and analyze vast amounts of text data. |
| hard | iterative | 621 | Deep reinforcement learning is how self-driving cars, drones, and advanced AI agents are trained. |
| hard | iterative | 786 | The narrator says each missing-data method provides a different type of value to the dataset. |
| hard | iterative | 874 | The people in the NYU canteen were not chosen by chance. |
| simple | baseline | 152 | C++ was created by Bjarn Strausdrup. |
| simple | full_context | 170 | The lecturer mentions Informix as a modern database. |
| simple | full_context | 170 | The lecturer mentions MySQL as a modern database. |
| simple | full_context | 170 | The lecturer mentions Oracle as a modern database. |
| simple | full_context | 170 | The lecturer mentions PostgreSQL as a modern database. |
| simple | full_context | 619 | NLP is described as the AI interpreter between humans and machines. |
| simple | full_context | 811 | The Future of Jobs report, 2023, highlights an expected 40% growth and demand for AI and machine learning specialists. |
| simple | iterative | 170 | The lecturer mentions MongoDB. |
| simple | iterative | 619 | NLP is described as the AI interpreter between humans and machines. |
| simple | iterative | 1005 | The advent of Tableau allowed BI analysts to be independent. |
| simple | iterative | 1124 | Anne shops from www.shoppingcart.com. |
| simple | iterative | 1245 | Large language models are instances of foundation models applied specifically to text and text-like things. |
| simple | iterative | 1404 | Dave asks what a quantum computer is. |
| very hard | baseline | 192 | As computers became faster, they could process punch card tasks quicker than the cards could be fed in. |
| very hard | baseline | 973 | Data integrity is described as a strong advantage when working with databases. |
| very hard | full_context | 43 | There are many supporting frameworks that make the machine learning process approachable. |
| very hard | iterative | 43 | There are many supporting frameworks out there to make the machine learning process approachable. |
| very hard | iterative | 75 | SQL is the gold standard language for communicating with relational database management systems. |
| very hard | iterative | 389 | Clustering is mentioned before dimensionality reduction in the sequence. |
| very hard | iterative | 628 | Natural Language Processing is described as the AI interpreter between humans and machines. |
| very hard | iterative | 940 | The median is the middle number in an ordered data set. |
| very hard | iterative | 973 | The database features mentioned improve data integrity. |

</details>

LLM cost (this run):

- `gpt-4.1-mini-2025-04-14`: 2260 calls (0 cached) · 667,683 in / 150,006 out · $0.5071
- run total: $0.5071
- cumulative (all recorded runs, `reports/llm_ledger.jsonl`): 33,483 calls · 28,568,878 in / 1,396,762 out · $22.05 · 6 row(s) unpriced · 14,412,524 tokens estimated
