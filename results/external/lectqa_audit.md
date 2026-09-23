# LectQA-Vid — benchmark audit

LLM-free. Published files: `mcq_questions.json`, `open_ended_questions.json`, `video_links.docx` (Mendeley doi:10.17632/yt4nmz9mcv.1). Obtained media: YouTube, 2026-09-23.

## (a) MCQ answer position

All 1489 published MCQs. The correct answer's position in the stored option list:

| position | count | share |
|---|---:|---:|
| A | 1484 | 99.7% |
| B | 4 | 0.3% |
| C | 1 | 0.1% |

**A constant-"A" answerer scores 99.7%** (hard 100.0%, simple 99.0%, very hard 100.0%), against their reported 53.43 %. Any MCQ run that keeps the stored order measures position preference, not retrieval. Options must be shuffled.

## (b) Gold timestamps

Every start/end stamp of all 2982 published QA pairs, by digit shape (9 = any digit):

| shape | stamps |
|---|---:|
| `99:99:99` | 4069 |
| `99:99` | 683 |
| `99:999:99` | 550 |
| `999:99` | 532 |
| `999.99` | 67 |
| `99:999` | 26 |
| `99:99:999` | 21 |
| `99.99` | 16 |

Unambiguous by `strict_seconds` (HH:MM:SS / MM:SS with fields < 60, or plain seconds): 4532 of 5964 stamps. `00:12:70` (= 12.70 s), `258:36` and `01:91:60` also occur, and their meaning differs between videos.

Gold intervals of the 2832 QA pairs of the 95 obtained videos: ambiguous format 878, beyond the fetched video's end 355, end before start 1, ok 1598.

| videos | scorable | ambiguous | past the video's end | other |
|---|---:|---:|---:|---:|
| 1–35 | 828 | 99 | 32 | 1 |
| 36–63 | 595 | 176 | 27 | 0 |
| 64–100 | 175 | 603 | 296 | 0 |

## (c) Link availability

95/100 obtained, 5 dead. Mendeley ships no media or transcripts, so every video comes from its YouTube link.

| video | reason |
|---|---|
| video_1 | ERROR: [youtube] BUVOnAVsUmk: This video is not available |
| video_10 | ERROR: [youtube] kQUG_S3dGF8: This video is not available |
| video_12 | ERROR: [youtube] KwhkzQAnzjU: This video is not available |
| video_38 | ERROR: unable to download video data: HTTP Error 403: Forbidden |
| video_75 | ERROR: unable to download video data: HTTP Error 403: Forbidden |

## (d) Genre

**Rule, stated before looking at answers:** slide changes per minute from the frame-derived deck (1 fps dHash segmentation). **≤ 3 per minute = slide talk** (a slide stays up ≥ 20 s), **> 6 = animated explainer**, otherwise mixed. The relatedness-gate verdict (does the speech align with the recovered deck) is reported next to it, not folded into the rule.

| class | gate accepted | gate rejected |
|---|---:|---:|
| slide talk | 3 | 16 |
| mixed | 18 | 11 |
| animated explainer | 25 | 22 |
| **total** | 46 | 49 |

<details><summary>per video</summary>

| video | duration s | slides | slides/min | gate | class |
|---|---:|---:|---:|---|---|
| video_2 | 99 | 20 | 12.1 | accepted | animated explainer |
| video_3 | 91 | 19 | 12.6 | rejected | animated explainer |
| video_4 | 154 | 36 | 14.0 | rejected | animated explainer |
| video_5 | 140 | 27 | 11.6 | rejected | animated explainer |
| video_6 | 142 | 25 | 10.5 | rejected | animated explainer |
| video_7 | 143 | 33 | 13.8 | accepted | animated explainer |
| video_8 | 286 | 32 | 6.7 | accepted | animated explainer |
| video_9 | 150 | 21 | 8.4 | rejected | animated explainer |
| video_11 | 139 | 22 | 9.5 | accepted | animated explainer |
| video_13 | 136 | 4 | 1.8 | rejected | slide talk |
| video_14 | 165 | 29 | 10.5 | rejected | animated explainer |
| video_15 | 234 | 28 | 7.2 | accepted | animated explainer |
| video_16 | 210 | 31 | 8.9 | rejected | animated explainer |
| video_17 | 99 | 3 | 1.8 | rejected | slide talk |
| video_18 | 109 | 5 | 2.8 | rejected | slide talk |
| video_19 | 68 | 15 | 13.2 | rejected | animated explainer |
| video_20 | 352 | 32 | 5.5 | accepted | mixed |
| video_21 | 111 | 15 | 8.1 | rejected | animated explainer |
| video_22 | 147 | 19 | 7.7 | rejected | animated explainer |
| video_23 | 328 | 43 | 7.9 | accepted | animated explainer |
| video_24 | 154 | 14 | 5.5 | rejected | mixed |
| video_25 | 59 | 10 | 10.1 | rejected | animated explainer |
| video_26 | 156 | 22 | 8.5 | accepted | animated explainer |
| video_27 | 139 | 22 | 9.5 | accepted | animated explainer |
| video_28 | 64 | 9 | 8.5 | accepted | animated explainer |
| video_29 | 301 | 24 | 4.8 | accepted | mixed |
| video_30 | 131 | 13 | 5.9 | accepted | mixed |
| video_31 | 264 | 14 | 3.2 | accepted | mixed |
| video_32 | 267 | 28 | 6.3 | rejected | animated explainer |
| video_33 | 257 | 33 | 7.7 | accepted | animated explainer |
| video_34 | 131 | 13 | 5.9 | accepted | mixed |
| video_35 | 330 | 33 | 6.0 | accepted | animated explainer |
| video_36 | 64 | 9 | 8.5 | accepted | animated explainer |
| video_37 | 302 | 34 | 6.8 | accepted | animated explainer |
| video_39 | 354 | 19 | 3.2 | rejected | mixed |
| video_40 | 159 | 16 | 6.0 | accepted | animated explainer |
| video_41 | 139 | 22 | 9.5 | accepted | animated explainer |
| video_42 | 72 | 9 | 7.5 | rejected | animated explainer |
| video_43 | 162 | 21 | 7.8 | rejected | animated explainer |
| video_44 | 128 | 7 | 3.3 | rejected | mixed |
| video_45 | 458 | 22 | 2.9 | accepted | slide talk |
| video_46 | 74 | 16 | 13.0 | rejected | animated explainer |
| video_47 | 227 | 7 | 1.8 | rejected | slide talk |
| video_48 | 87 | 11 | 7.6 | rejected | animated explainer |
| video_49 | 263 | 18 | 4.1 | rejected | mixed |
| video_50 | 159 | 31 | 11.7 | accepted | animated explainer |
| video_51 | 304 | 32 | 6.3 | accepted | animated explainer |
| video_52 | 304 | 46 | 9.1 | accepted | animated explainer |
| video_53 | 88 | 7 | 4.8 | rejected | mixed |
| video_54 | 137 | 21 | 9.2 | rejected | animated explainer |
| video_55 | 272 | 11 | 2.4 | rejected | slide talk |
| video_56 | 261 | 15 | 3.5 | accepted | mixed |
| video_57 | 341 | 21 | 3.7 | accepted | mixed |
| video_58 | 398 | 35 | 5.3 | rejected | mixed |
| video_59 | 489 | 44 | 5.4 | accepted | mixed |
| video_60 | 328 | 37 | 6.8 | rejected | animated explainer |
| video_61 | 326 | 43 | 7.9 | rejected | animated explainer |
| video_62 | 475 | 28 | 3.5 | accepted | mixed |
| video_63 | 233 | 12 | 3.1 | accepted | mixed |
| video_64 | 210 | 12 | 3.4 | accepted | mixed |
| video_65 | 377 | 67 | 10.7 | accepted | animated explainer |
| video_66 | 308 | 33 | 6.4 | accepted | animated explainer |
| video_67 | 264 | 12 | 2.7 | rejected | slide talk |
| video_68 | 168 | 7 | 2.5 | accepted | slide talk |
| video_69 | 426 | 6 | 0.8 | rejected | slide talk |
| video_70 | 291 | 3 | 0.6 | rejected | slide talk |
| video_71 | 257 | 9 | 2.1 | rejected | slide talk |
| video_72 | 264 | 16 | 3.6 | rejected | mixed |
| video_73 | 433 | 9 | 1.2 | rejected | slide talk |
| video_74 | 248 | 9 | 2.2 | rejected | slide talk |
| video_76 | 212 | 8 | 2.3 | rejected | slide talk |
| video_77 | 312 | 38 | 7.3 | rejected | animated explainer |
| video_78 | 329 | 36 | 6.6 | rejected | animated explainer |
| video_79 | 391 | 43 | 6.6 | rejected | animated explainer |
| video_80 | 278 | 35 | 7.6 | accepted | animated explainer |
| video_81 | 427 | 44 | 6.2 | accepted | animated explainer |
| video_82 | 338 | 27 | 4.8 | accepted | mixed |
| video_83 | 407 | 37 | 5.5 | accepted | mixed |
| video_84 | 359 | 34 | 5.7 | accepted | mixed |
| video_85 | 349 | 28 | 4.8 | accepted | mixed |
| video_86 | 324 | 34 | 6.3 | accepted | animated explainer |
| video_87 | 322 | 28 | 5.2 | accepted | mixed |
| video_88 | 416 | 24 | 3.5 | rejected | mixed |
| video_89 | 333 | 16 | 2.9 | rejected | slide talk |
| video_90 | 313 | 16 | 3.1 | rejected | mixed |
| video_91 | 380 | 17 | 2.7 | rejected | slide talk |
| video_92 | 323 | 9 | 1.7 | rejected | slide talk |
| video_93 | 266 | 38 | 8.6 | accepted | animated explainer |
| video_94 | 379 | 20 | 3.2 | rejected | mixed |
| video_95 | 481 | 32 | 4.0 | accepted | mixed |
| video_96 | 498 | 48 | 5.8 | rejected | mixed |
| video_97 | 370 | 32 | 5.2 | accepted | mixed |
| video_98 | 458 | 8 | 1.0 | rejected | slide talk |
| video_99 | 496 | 15 | 1.8 | accepted | slide talk |
| video_100 | 299 | 37 | 7.4 | accepted | animated explainer |

</details>

## (e) Context fit

Full transcript per obtained video, o200k tokens (GPT-4.1 / GPT-5 encoding), n = 95: min 140, median 780, max 1544.

| window | videos whose whole transcript fits |
|---|---:|
| 2,048 | 95/95 |
| 4,096 | 95/95 |
| 8,192 | 95/95 |
| 32,768 | 95/95 |

The API does not report the answerer's (gpt-5.4-mini-2026-03-17) context window. The largest transcript is far below any current OpenAI window, so a no-retrieval full-transcript answerer is a valid baseline on this benchmark.

