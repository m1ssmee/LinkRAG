"""Fill the paper's XFile results (Table 3 and Figure 5) from compare_retrieval's per-question results.

    python paper/xfile_fill.py results/xfile/<test-half>.questions.json      # writes Table 3 + paper/xfile_results.json
    python paper/figures.py                                                  # draws Figure 5 from paper/xfile_results.json

Input is the `<out>.questions.json` that `compare_retrieval.py --correctness --out <out>` writes for
the test half (gp_scaling, mimic_data, rl_mdp). Paired differences and their 95 % intervals come from
`scripts/adapters/xfile.py paired` (topics resampled), so the paper and the benchmark report use one
computation. Only Table 3 and Figure 5 are generated; the sentences around them are written by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))
from xfile import paired  # noqa: E402

MANIFEST = "a086ca1c22dc1d93"
TEST_TOPICS = {"gp_scaling", "mimic_data", "rl_mdp"}
LINKED, CONTROL = "linkrag|complementarity", "baseline|complementarity"
ROWS = [("Hybrid retrieval, top-k", "baseline|none"),
        ("Hybrid + link expansion, top-k", "linkrag|none"),
        ("No-links control + complementarity", CONTROL),
        ("LinkRAG (links + complementarity)", LINKED)]
# The test half holds a single deictic question (MD10), so deictic is counted inside cross-file and
# never gets a column or interval of its own.
GROUPS = [("Cross-file", {"cross_modal_split", "cross_modal_deictic"}),
          ("Single-source", {"audio_only", "slides_only", "paper_only"}),
          ("Overall", None)]


def interval(r: dict) -> str:
    """A bootstrap over fewer than two topics has nothing to resample: no interval is shown."""
    if r["topics"] < 2:
        return f"{signed(r['diff'])} (one topic, no interval)"
    return f"{signed(r['diff'])} [{signed(r['lo'])}, {signed(r['hi'])}]"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def signed(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "−")


def check(d: dict) -> None:
    if d.get("corpus") != MANIFEST:
        sys.exit(f"corpus {d.get('corpus')} is not the XFile manifest {MANIFEST}")
    topics = {q["topic"] for q in d["questions"]}
    if topics != TEST_TOPICS:
        sys.exit(f"expected the test half {sorted(TEST_TOPICS)}, got {sorted(topics)}")
    for q in d["questions"]:
        for _, cell in ROWS:
            runs = d["correct"].get(f"{q['qid']}|{cell}")
            if not runs or len(runs) < 3:
                sys.exit(f"{q['qid']} {cell}: {len(runs or [])} runs, need 3")


def table(path: Path, d: dict) -> tuple[str, dict]:
    head = "".join(f'<th class="n">{g}</th>' for g, _ in GROUPS)
    n = {g: sum(1 for q in d["questions"] if t is None or q["type"] in t) for g, t in GROUPS}
    n_deictic = sum(1 for q in d["questions"] if q["type"] == "cross_modal_deictic")
    out = ['<table id="tab-xfile">',
           f'<caption><b>Table 3.</b> Answer correctness (%) on the XFile test half ({", ".join(sorted(TEST_TOPICS))}), '
           f'mean over three repeats; n = {n["Overall"]} questions ({n["Cross-file"]} cross-file, of which {n_deictic} deictic; '
           f'{n["Single-source"]} single-source). The last row is the paired '
           'difference between LinkRAG and the no-links control, with a 95 % interval from resampling topics.</caption>',
           f'<tr class="h"><th>Method</th>{head}</tr>']
    stats = {}
    for name, cell in ROWS:
        tds = ""
        for g, types in GROUPS:
            r = paired(path, cell, cell, types=types)       # a == b: just the cell mean over its questions
            tds += f'<td class="n">{pct(r["a"])}</td>' if r["n"] else '<td class="n">—</td>'
        cls = ' class="key"' if cell == LINKED else ""
        out.append(f"<tr{cls}><td>{name}</td>{tds}</tr>")
    tds = ""
    for g, types in GROUPS:
        r = paired(path, LINKED, CONTROL, types=types)
        stats[g] = r
        tds += f'<td class="n">{interval(r)}</td>'
    out.append(f'<tr class="rule last"><td>LinkRAG − no-links control</td>{tds}</tr>')
    out.append("</table>")
    return "\n".join(out), stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, help="test-half <out>.questions.json from compare_retrieval.py")
    ap.add_argument("--paper", type=Path, default=ROOT / "paper" / "paper.html")
    ap.add_argument("--json", type=Path, default=ROOT / "paper" / "xfile_results.json")
    args = ap.parse_args(argv)

    d = json.loads(args.results.read_text())
    check(d)
    html, stats = table(args.results, d)
    t = args.paper.read_text()
    pat = re.compile(r'<table( id="tab-xfile")?>\s*<caption><b>Table 3\.</b>.*?</table>', re.S)
    if len(pat.findall(t)) != 1:
        sys.exit("expected exactly one Table 3 in the paper")
    args.paper.write_text(pat.sub(lambda _m: html, t))
    args.json.write_text(json.dumps({
        "source": str(args.results), "corpus": d["corpus"], "a": LINKED, "b": CONTROL,
        "groups": [{"name": g, **{k: stats[g][k] for k in ("n", "topics", "a", "b", "diff", "lo", "hi", "verdict")}}
                   for g, _ in GROUPS]}, indent=1))
    for g, _ in GROUPS:
        r = stats[g]
        print(f"{g:<22} n={r['n']:<3} LinkRAG {pct(r['a']):>5}  control {pct(r['b']):>5}  "
              f"diff {interval(r)}  {r['verdict'] if r['topics'] >= 2 else 'no interval'}")
    print(f"Table 3 written to {args.paper}; figure data to {args.json}. Now run: python paper/figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
