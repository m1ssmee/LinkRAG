# Judge agreement (offline): claude_agents_judge:claude-sonnet-5-5, pilot01

Candidate judge `claude_agents_judge:claude-sonnet-5-5`, 3 run(s) per prompt, against the stored verdicts (majority of 3) in `reports/gold_verified_pilot01.json` (gpt-4.1-mini) and `.judge-gpt54.json` (gpt-5.4). Prompts rebuilt from the stored unit texts and answers with verify_gold's `entail_unit` / `grade` (`scripts/eval/judge_agreement_offline.py`). Bar, set before the run: unit kappa vs gpt-4.1-mini >= 0.7.

| comparison | n | agreement | Cohen's kappa |
|---|---|---|---|
| unit kept vs gpt-4.1-mini | 147 | 133/147 | 0.81 |
| unit kept vs gpt-5.4 | 147 | 142/147 | 0.93 |
| *reference: gpt-4.1-mini vs gpt-5.4* | 147 | 136/147 | 0.85 |
| grade PASS vs gpt-4.1-mini | 100 | 98/100 | 0.96 |
| type label vs gpt-4.1-mini | 25 | 24/25 | - |

**Verdict: PASSES the bar** (unit kappa vs gpt-4.1-mini 0.81, bar 0.7).

## Unit disagreements with gpt-4.1-mini

| qid | unit | candidate | gpt-4.1-mini | gpt-5.4 |
|---|---|---|---|---|
| Q4 | `osdi18_slides_hsieh:p20:g0` | True | False | False |
| Q4 | `osdi18-hsieh:p3:t2` | False | True | False |
| Q4 | `osdi18-hsieh:p9:t1` | False | True | True |
| Q4 | `osdi18-hsieh:p13:t3` | True | False | False |
| A8 | `osdi18-hsieh:p9:t1` | True | False | True |
| A8 | `osdi18-hsieh:p9:c0` | False | True | False |
| B2 | `osdi18-hsieh:p7:t4` | True | False | True |
| B5 | `osdi18_slides_hsieh:p20:t0` | True | False | True |
| B5 | `osdi18_slides_hsieh:p20:g0` | True | False | True |
| B6 | `osdi18-hsieh:p12:t2` | True | False | True |
| C2 | `osdi18_slides_hsieh:p13:t0` | True | False | True |
| C3 | `osdi18_slides_hsieh:p14:t0` | True | False | True |
| C5 | `osdi18_slides_hsieh:p22:g0` | False | True | True |
| C7 | `osdi18_slides_hsieh:p23:t0` | True | False | True |

## Type-label disagreements

| qid | candidate | stored |
|---|---|---|
| C3 | single_modality | audio_only |
