"""Build the paper's inline SVG figures and insert them into paper/paper.html (idempotent).

    python paper/figures.py && python scripts/render_pdf.py paper/paper.html paper/LinkRAG_paper.pdf

paper.html holds one <figure id="figN">…</figure> per figure; this script replaces each one whole.
Numbers are copied from committed sources:
  fig1  tests/regression/xfile_questions.jsonl, results/xfile/gold_check_3runs.md, reports/xfile_review.csv
  fig3  data/processed/xfile/links.jsonl (unit ids and scores also in results/xfile/links.jsonl)
  fig4  build_links log, audio_slide gate lines (DESIGN.md known issue (n))
  fig5  schematic of Eqs. 7-8 (illustrative units, no data)
  fig6  schematic of the four XFile conditions (no data)
  fig7  paper/xfile_results.json (paper/xfile_fill.py, from results/xfile/eval_test.questions.json)
  fig8  results/external/lectqa_recheck.md
  fig9  results/external/mavils_recheck.md
"""
import re

INK, MUTED, RULE, BG = "#1d232b", "#5b6270", "#c9ced6", "#fff"
# one colour per stage: sources/indexing, the proposed linking components, question time, answer
STAGE = {
    "src":  ("#0f766e", "#e7f4f2"),
    "link": ("#1d5fbf", "#e8f0fc"),
    "qry":  ("#4b5563", "#f3f4f6"),
    "ans":  ("#b45309", "#fdf2e4"),
}
BLUE, ORANGE, GREY = "#1d5fbf", "#d9622b", "#9aa1ab"
F = 'font-family="Helvetica, Arial, sans-serif"'


def svg(w, h, body):
    defs = ('<defs>'
            f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
            f'<marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>'
            f'<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker>'
            '</defs>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">{defs}{body}</svg>'


def text(x, y, s, size=9.5, anchor="start", fill=INK, weight="normal", style="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {F} font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}" font-style="{style}">{s}</text>')


def badge(x, y, n, colour):
    return (f'<circle cx="{x}" cy="{y}" r="8" fill="{colour}"/>'
            + text(x, y + 3.4, n, 9.5, "middle", "#fff", "700"))


def card(x, y, w, h, title, lines, stage, n=None, proposed=False, size=8.6):
    stroke, fill = STAGE[stage]
    dash = ' stroke-dasharray="5 3"' if proposed else ""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{1.6 if proposed else 1.1}"{dash}/>'
    tx = x + (24 if n else 9)
    if n:
        out += badge(x + 13, y + 14, n, stroke)
    out += text(tx, y + 18, title, 9.6, fill=INK, weight="700")
    for i, l in enumerate(lines):
        out += text(x + 9, y + 33 + i * 11.5, l, size, fill=MUTED)
    return out


def line(x1, y1, x2, y2, colour=INK, dashed=False, marker="ah", width=1.2):
    d = ' stroke-dasharray="5 3"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="{width}"{d} marker-end="url(#{marker})"/>'


def path(d, colour=INK, dashed=False, marker="ah", width=1.2):
    da = ' stroke-dasharray="5 3"' if dashed else ""
    return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{da} marker-end="url(#{marker})"/>'


# ---------------------------------------------------------------- figure 1: how XFile was built
def fig1():
    b = text(0, 13, "(a) From drafted questions to verified gold", 10, weight="700")
    steps = [("1", "Drafted", 98, "7 topics × 14 questions", "src"),
             ("2", "Human review", 90, "8 rejected: unrelated facts joined", "qry"),
             ("3", "Gold check, 3 runs", 86, "4 dropped, 6 relabelled to one source", "link"),
             ("4", "Verified cross-file", 53, "44 split across files + 9 deictic", "ans")]
    L, R, top, rh = 150, 440, 26, 34
    sx = lambda v: L + (R - L) * v / 98
    for i, (n, name, v, note, st) in enumerate(steps):
        y = top + i * rh
        stroke, fill = STAGE[st]
        b += badge(10, y + 12, n, stroke) + text(24, y + 16, name, 9.6, weight="700")
        b += f'<rect x="{L}" y="{y + 2}" width="{sx(v) - L:.1f}" height="20" rx="3" fill="{stroke}" fill-opacity="0.35" stroke="{stroke}" stroke-width="1.1"/>'
        b += text(sx(v) + 6, y + 16, str(v), 10, weight="700")
        b += text(sx(v) + 30, y + 16, note, 8.4, fill=MUTED)
        if i:
            b += line(L + 10, y - 10, L + 10, y + 1, GREY, marker="ag", width=1)
    y0 = top + 4 * rh + 26
    b += text(0, y0, "(b) The 86 verified questions by split: cross-file vs single-source", 10, weight="700")
    rows = [("Tune: nfa_closure, psych_methods, visual_system", 23, 13), ("Test: gp_scaling, mimic_data, rl_mdp", 21, 17),
            ("Separate row: lightning (intake gate REJECT)", 9, 3)]
    L2, unit = 260, 9.0
    for i, (name, cross, single) in enumerate(rows):
        y = y0 + 12 + i * 24
        b += text(L2 - 8, y + 13, name, 8.8, "end", fill=INK if i < 2 else MUTED, style="normal" if i < 2 else "italic")
        b += f'<rect x="{L2}" y="{y + 2}" width="{cross * unit}" height="16" fill="{STAGE["link"][0]}"/>'
        b += text(L2 + cross * unit / 2, y + 13.5, cross, 8.8, "middle", "#fff", "700")
        b += f'<rect x="{L2 + cross * unit}" y="{y + 2}" width="{single * unit}" height="16" fill="{GREY}"/>'
        b += text(L2 + cross * unit + single * unit / 2, y + 13.5, single, 8.8, "middle", "#fff", "700")
    ly = y0 + 12 + 3 * 24 + 8
    b += f'<rect x="{L2}" y="{ly}" width="10" height="10" fill="{STAGE["link"][0]}"/>' + text(L2 + 15, ly + 9, "cross-file (needs two or more files)", 8.6)
    b += f'<rect x="{L2 + 200}" y="{ly}" width="10" height="10" fill="{GREY}"/>' + text(L2 + 215, ly + 9, "single-source control", 8.6)
    return svg(680, ly + 16, b)


# ---------------------------------------------------------------- figure 2: the proposed pipeline
def fig2():
    b = text(0, 12, "INDEXING TIME", 8.8, fill=MUTED, weight="700")
    y1, h = 22, 70
    top = [(0, 142, "1", "Sources", ["lecture recording,", "slide deck,", "notes / paper PDF"], "src", False),
           (162, 142, "2", "Evidence units", ["speech segment, text", "block or figure, each", "with page or time span"], "src", False),
           (324, 142, "3", "Hybrid index", ["BGE-M3 dense vectors", "+ BM25 sparse index"], "src", False),
           (486, 194, "4", "Evidence Linking Layer", ["audio–slide · figure–text ·", "deictic · same-slide links,", "relatedness gate per file pair"], "link", True)]
    for i, (x, w, n, t, ls, st, pr) in enumerate(top):
        b += card(x, y1, w, h, t, ls, st, n, pr)
        b += TOP_ICON[i](x + w - 8, y1 + 6)
        if i:
            px, pw = top[i - 1][0], top[i - 1][1]
            b += line(px + pw + 1, y1 + h / 2, x - 2, y1 + h / 2)
    y2, w2, h2 = 168, 110, 92
    b += text(0, y2 - 10, "QUESTION TIME", 8.8, fill=MUTED, weight="700")
    b += f'<rect x="0" y="{y2 + h2 / 2 - 13}" width="62" height="26" rx="13" fill="#fff" stroke="{INK}" stroke-width="1.1"/>' + text(31, y2 + h2 / 2 + 4, "Question", 9.2, "middle", weight="700")
    bot = [(82, "5", "Hybrid retrieval", ["dense + BM25,", "rank fusion"], "qry", False),
           (204, "6", "Link expansion", ["add 1-hop linked", "neighbours of seeds"], "link", True),
           (326, "7", "Set reranker", ["coverage + links", "− redundancy"], "link", True),
           (448, "8", "Answerer", ["claims, each citing", "evidence units"], "ans", False),
           (570, "9", "Verifier", ["each claim checked", "against its citation"], "ans", False)]
    b += line(62, y2 + h2 / 2, 80, y2 + h2 / 2)
    for i, (x, n, t, ls, st, pr) in enumerate(bot):
        b += card(x, y2, w2, h2, t, ls, st, n, pr)
        b += BOT_ICON[i](x + w2 / 2, y2 + 58)
        if i:
            b += line(bot[i - 1][0] + w2 + 1, y2 + h2 / 2, x - 2, y2 + h2 / 2)
    # index feeds retrieval; links feed expansion and reranking
    b += path(f"M {324 + 71} {y1 + h} V {y1 + h + 22} H {82 + 55} V {y2 - 2}")
    b += path(f"M {486 + 70} {y1 + h} V {y1 + h + 34} H {204 + 55} V {y2 - 2}", BLUE, True, "ab")
    b += path(f"M {486 + 124} {y1 + h} V {y1 + h + 34} H {326 + 55} V {y2 - 2}", BLUE, True, "ab")
    b += text(486 + 97, y1 + h + 46, "typed, scored links", 8.4, "middle", BLUE, style="italic")
    # optional re-query loop
    b += path(f"M {448 + 55} {y2 + h2} V {y2 + h2 + 20} H {82 + 55} V {y2 + h2 + 2}", GREY, True, "ag")
    b += text((448 + 82 + 110) / 2, y2 + h2 + 32, "optional: one follow-up query written by the answering model", 8.4, "middle", MUTED, style="italic")
    ly = y2 + h2 + 50
    b += f'<rect x="0" y="{ly}" width="22" height="12" rx="3" fill="{STAGE["link"][1]}" stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="5 3"/>' + text(28, ly + 10, "proposed component", 8.6)
    for i, (k, lab) in enumerate([("src", "indexing"), ("qry", "retrieval"), ("ans", "answer and verification")]):
        x = 170 + i * 140
        b += f'<rect x="{x}" y="{ly}" width="22" height="12" rx="3" fill="{STAGE[k][1]}" stroke="{STAGE[k][0]}" stroke-width="1.1"/>' + text(x + 28, ly + 10, lab, 8.6)
    return svg(680, ly + 16, b)


# ---------------------------------------------------------------- figure 3: links on a real example
def fig3():
    C1, C2, C3, w = 0, 250, 500, 180
    b = ""
    for x, t_ in ((C1, "LECTURE RECORDING"), (C2, "SLIDE DECK"), (C3, "TEXTBOOK CHAPTER")):
        b += text(x, 12, t_, 8.8, fill=MUTED, weight="700")
    b = text(0, 12, "Each blue arrow is one link the layer built; the number under it is the link's score, from 0 (weak) to 1 (certain).", 8.8, fill=INK) + f'<g transform="translate(0,24)">' + b
    g1, g2 = (C1 + w + C2) / 2, (C2 + w + C3) / 2
    b += card(C1, 22, w, 82, "Speech, 06:40", ["“…a sequence of random", "states S1, S2, … with the Markov", "property. That's the definition", "of a Markov process…”"], "qry", size=8.4)
    b += card(C2, 22, w, 82, "Slide 6, text block", ["“Markov Process: a memoryless", "random process, i.e. a sequence", "of random states S1, S2, … with", "the Markov property.”"], "qry", size=8.4)
    b += card(C2, 140, w, 64, "Slide 6, figure", ["definition box: tuple ⟨S, P⟩", "and transition matrix P(ss′)"], "qry", size=8.4)
    b += card(C3, 140, w, 64, "Ch. 3 paragraph", ["recycling-robot example:", "transition probabilities α, β"], "qry", size=8.4)
    b += ic_wave(C1 + w - 52, 30, 44, 12, n=12) + ic_slide(C2 + w - 34, 28, 26, 18) + ic_chart(C2 + w - 32, 148, 24, 16) + ic_doc(C3 + w - 24, 146, 16, 20)
    b += line(C1 + w + 2, 63, C2 - 2, 63, BLUE, marker="ab", width=1.8) + badge(g1, 47, "1", BLUE)
    b += text(g1, 80, "audio–slide link", 8.4, "middle", BLUE, "700") + text(g1, 91, "score 0.81", 8.4, "middle", BLUE)
    b += line(C2 + w / 2, 138, C2 + w / 2, 106, BLUE, marker="ab", width=1.8) + badge(C2 + w / 2 - 16, 122, "2", BLUE)
    b += text(C2 + w / 2 + 8, 121, "same-slide link", 8.4, fill=BLUE, weight="700") + text(C2 + w / 2 + 8, 132, "score 1.00", 8.4, fill=BLUE)
    b += line(C2 + w + 2, 172, C3 - 2, 172, BLUE, marker="ab", width=1.8) + badge(g2, 156, "3", BLUE)
    b += text(g2, 189, "figure–text link", 8.4, "middle", BLUE, "700") + text(g2, 200, "score 0.60", 8.4, "middle", BLUE)
    y = 226
    b += f'<line x1="0" y1="{y - 8}" x2="680" y2="{y - 8}" stroke="{RULE}" stroke-width="0.8"/>'
    b += card(C1, y, w, 64, "Speech, 01:26 (Lightning)", ["“…now Bob can use Bob's key", "and this revocable key…”"], "qry", size=8.4)
    b += card(C2, y, w, 64, "Slide 4, figure", ["diagram labelled", "“Commit Tx (held by Bob)”"], "qry", size=8.4)
    b += ic_wave(C1 + w - 52, y + 46, 44, 12, n=12) + ic_chart(C2 + w - 32, y + 8, 24, 16)
    b += line(C1 + w + 2, y + 32, C2 - 2, y + 32, BLUE, marker="ab", width=1.8) + badge(g1, y + 16, "4", BLUE)
    b += text(g1, y + 49, "deictic link", 8.4, "middle", BLUE, "700") + text(g1, y + 60, "score 0.65", 8.4, "middle", BLUE)
    b += text(C3, y + 22, "A pointing phrase (“this …”) is", 8.6, fill=MUTED)
    b += text(C3, y + 35, "linked to the visual element on", 8.6, fill=MUTED)
    b += text(C3, y + 48, "the slide on screen at that moment.", 8.6, fill=MUTED)
    return svg(680, y + 70 + 24, b + "</g>")


# ---------------------------------------------------------------- figure 4: the relatedness gate on XFile
GATE = [("psych_methods","rl_mdp",-0.6), ("psych_methods","gp_scaling",-0.6), ("psych_methods","nfa_closure",-0.7), ("psych_methods","mimic_data",-0.1), ("psych_methods","psych_methods",12.2), ("psych_methods","visual_system",1.4), ("psych_methods","lightning",-0.2), ("visual_system","rl_mdp",-0.3), ("visual_system","gp_scaling",-0.4), ("visual_system","nfa_closure",-0.2), ("visual_system","mimic_data",-0.2), ("visual_system","psych_methods",-0.2), ("visual_system","visual_system",-0.1), ("visual_system","lightning",-0.1), ("lightning","rl_mdp",0.0), ("lightning","gp_scaling",-0.8), ("lightning","nfa_closure",-0.7), ("lightning","mimic_data",-1.0), ("lightning","psych_methods",2.6), ("lightning","visual_system",0.7), ("lightning","lightning",4.9), ("rl_mdp","rl_mdp",3.9), ("rl_mdp","gp_scaling",-1.6), ("rl_mdp","nfa_closure",1.4), ("rl_mdp","mimic_data",-0.0), ("rl_mdp","psych_methods",-0.2), ("rl_mdp","visual_system",-0.0), ("rl_mdp","lightning",-0.3), ("gp_scaling","rl_mdp",-0.2), ("gp_scaling","gp_scaling",8.0), ("gp_scaling","nfa_closure",2.1), ("gp_scaling","mimic_data",-0.2), ("gp_scaling","psych_methods",0.2), ("gp_scaling","visual_system",-0.0), ("gp_scaling","lightning",-0.5), ("mimic_data","rl_mdp",-0.3), ("mimic_data","gp_scaling",-0.4), ("mimic_data","nfa_closure",-0.4), ("mimic_data","mimic_data",-1.5), ("mimic_data","psych_methods",-0.1), ("mimic_data","visual_system",0.2), ("mimic_data","lightning",0.3), ("nfa_closure","rl_mdp",-0.3), ("nfa_closure","gp_scaling",-0.7), ("nfa_closure","nfa_closure",2.3), ("nfa_closure","mimic_data",-0.0), ("nfa_closure","psych_methods",0.5), ("nfa_closure","visual_system",0.3), ("nfa_closure","lightning",0.7)]
SHORT = {"psych_methods": "psych", "visual_system": "visual", "lightning": "lightning", "rl_mdp": "rl", "gp_scaling": "gp", "mimic_data": "mimic", "nfa_closure": "nfa"}


def fig4():
    Z = 1.27
    L, R = 170, 660
    lo, hi = -2.0, 13.0
    sx = lambda z: L + (R - L) * (z - lo) / (hi - lo)
    rows = [("Recording vs its own deck", 7, 60, [g for g in GATE if g[0] == g[1]]),
            ("Recording vs another lecture's deck", 42, 140, [g for g in GATE if g[0] != g[1]])]
    b = ""
    axis_y = 190
    for t in range(-2, 14, 2):
        b += f'<line x1="{sx(t):.1f}" y1="22" x2="{sx(t):.1f}" y2="{axis_y}" stroke="{RULE}" stroke-width="0.5"/>' + text(sx(t), axis_y + 13, t, 8.4, "middle", MUTED)
    b += text((L + R) / 2, axis_y + 27, "z: how much better a recording aligns with this deck than with the same deck shuffled (Eq. 2)", 8.6, "middle", MUTED)
    b += f'<rect x="{sx(Z):.1f}" y="18" width="{R - sx(Z):.1f}" height="{axis_y - 18}" fill="{STAGE["link"][1]}" opacity="0.55"/>'
    b += f'<line x1="{sx(Z):.1f}" y1="14" x2="{sx(Z):.1f}" y2="{axis_y}" stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="5 3"/>'
    b += text(sx(Z) + 5, 14, "admitted: links are built →", 8.6, fill=BLUE, weight="700") + text(sx(Z) - 5, 14, "← rejected: no links", 8.6, "end", MUTED, "700") + text(sx(Z), axis_y + 13, "τ", 9, "middle", BLUE, "700")
    for name, n, yc, pts in rows:
        b += text(L - 10, yc + 3, name, 9, "end", weight="700") + text(L - 10, yc + 15, f"{n} pairs · should be " + ("admitted" if n == 7 else "rejected"), 8.4, "end", MUTED)
        placed, false_acc = [], []
        for a, d, z in sorted(pts, key=lambda g: g[2]):
            x = sx(z)
            for k in range(0, 60):
                off = (k + 1) // 2 * 6 * (1 if k % 2 else -1)
                if all(abs(x - px) > 6.5 or off != po for px, po in placed):
                    break
            placed.append((x, off))
            same, admitted = a == d, z >= Z
            cy = yc + off
            if same and admitted:
                b += f'<circle cx="{x:.1f}" cy="{cy}" r="4.6" fill="{BLUE}"/>' + text(x, cy - 8, SHORT[a], 7.8, "middle")
            elif same:
                b += f'<circle cx="{x:.1f}" cy="{cy}" r="4.2" fill="#fff" stroke="{ORANGE}" stroke-width="1.8"/>' + text(x, cy - 8, SHORT[a], 7.8, "middle", ORANGE)
            elif admitted:
                b += f'<circle cx="{x:.1f}" cy="{cy}" r="4" fill="{ORANGE}"/>'
                false_acc.append((x, cy, f"{SHORT[a]} → {SHORT[d]} ({z:.1f})"))
            else:
                b += f'<circle cx="{x:.1f}" cy="{cy}" r="2.8" fill="#fff" stroke="{GREY}" stroke-width="1.1"/>'
        lx = sx(5.2)
        for i, (x, cy, lab) in enumerate(sorted(false_acc, key=lambda f: f[1])):
            ty = yc - 22 + i * 13
            b += f'<line x1="{x + 4:.1f}" y1="{cy}" x2="{lx - 3:.1f}" y2="{ty - 3}" stroke="{ORANGE}" stroke-width="0.7"/>'
            b += text(lx, ty, lab, 8, fill=ORANGE)
    ly = axis_y + 40
    items = [("full", BLUE, "correct: own deck linked (5)"), ("hollow", ORANGE, "error: own deck missed (2)"),
             ("full", ORANGE, "error: wrong lecture linked (4)"), ("hollow", GREY, "correct: other deck rejected (38)")]
    for i, (kind, c, lab) in enumerate(items):
        x = L - 160 + i * 168
        b += (f'<circle cx="{x + 5}" cy="{ly}" r="4.4" fill="{c}"/>' if kind == "full"
              else f'<circle cx="{x + 5}" cy="{ly}" r="4.2" fill="#fff" stroke="{c}" stroke-width="1.6"/>')
        b += text(x + 14, ly + 3.4, lab, 8.6)
    return svg(680, ly + 10, b)


# ---------------------------------------------------------------- figure 7: XFile paired differences
def fig7():
    """Forest plot of LinkRAG minus the no-links control, from paper/xfile_results.json (paper/xfile_fill.py)."""
    import json, os
    src = os.environ.get("XFILE_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "xfile_results.json")
    if not os.path.exists(src):
        b = f'<rect x="1" y="1" width="678" height="118" rx="6" fill="#fbf4f4" stroke="#8a1c1c" stroke-width="1" stroke-dasharray="6 4"/>'
        b += text(340, 52, "[[XFILE: run paper/xfile_fill.py on the test-half results]]", 10, "middle", "#8a1c1c", "700")
        b += text(340, 72, "paired difference in answer correctness, LinkRAG vs no-links control, by question type, 95 % CI over topics", 8.8, "middle", MUTED)
        return svg(680, 120, b)
    comps = json.load(open(src))["comparisons"]
    allg = [g for c in comps for g in c["groups"]]
    lo_ax = min(-0.3, min(g["lo"] for g in allg) - 0.05)
    hi_ax = max(0.3, max(g["hi"] for g in allg) + 0.05)
    L, R, top, rh = 190, 560, 34, 22
    sx = lambda v: L + (R - L) * (v - lo_ax) / (hi_ax - lo_ax)
    names = [g["name"] for g in comps[0]["groups"]]
    n_rows = len(names) * len(comps) + len(names) - 1
    bottom = top + rh * n_rows
    b = text(R + 115, top - 16, "difference [95 % CI]", 8.4, "end", MUTED)
    ticks = [x / 10 for x in range(int(lo_ax * 10), int(hi_ax * 10) + 1)]
    for v in ticks:
        b += f'<line x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{RULE}" stroke-width="0.5"/>'
        b += text(sx(v), bottom + 13, f"{100 * v:+.0f}".replace("-", "−").replace("+0", "0"), 8.4, "middle", MUTED)
    b += f'<line x1="{sx(0):.1f}" y1="{top - 8}" x2="{sx(0):.1f}" y2="{bottom}" stroke="{INK}" stroke-width="1"/>'
    shapes = [("rect", BLUE), ("circle", STAGE["src"][0])]
    row = 0
    for gi, name in enumerate(names):
        for ci, c in enumerate(comps):
            g = c["groups"][gi]
            y = top + row * rh + rh / 2
            if ci == 0:
                b += text(L - 10, y + 5, name, 9.4, "end", weight="700")
                b += text(L - 10, y + 16, f"n = {g['n']}", 8, "end", MUTED)
            shape, col = shapes[ci]
            b += f'<line x1="{sx(g["lo"]):.1f}" y1="{y}" x2="{sx(g["hi"]):.1f}" y2="{y}" stroke="{col}" stroke-width="2.4"/>'
            if shape == "rect":
                b += f'<rect x="{sx(g["diff"]) - 4.5:.1f}" y="{y - 4.5}" width="9" height="9" fill="{col}"/>'
            else:
                b += f'<circle cx="{sx(g["diff"]):.1f}" cy="{y}" r="4.8" fill="{col}"/>'
            lab = f"{100 * g['diff']:+.1f} [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}]".replace("-", "−").replace("+0.0", "0.0").replace("−0.0", "0.0")
            if max(abs(g["diff"]), abs(g["lo"]), abs(g["hi"])) < 1e-9:
                lab = "identical answers"
            b += text(R + 115, y + 3, lab, 8.8, "end", fill=col)
            row += 1
        row += 1
    ly = bottom + 30
    b += text((L + R) / 2, ly, "LinkRAG minus comparison, answer correctness (percentage points)", 8.6, "middle", MUTED)
    lx = L
    for (shape, col), c in zip(shapes, comps):
        b += (f'<rect x="{lx}" y="{ly + 10}" width="9" height="9" fill="{col}"/>' if shape == "rect"
              else f'<circle cx="{lx + 4.5}" cy="{ly + 14.5}" r="4.8" fill="{col}"/>')
        b += text(lx + 15, ly + 18, "LinkRAG " + c["label"], 8.8)
        lx += 200
    b += text(sx(0) - 6, top - 16, "← comparison better", 8.2, "end", MUTED) + text(sx(0) + 6, top - 16, "LinkRAG better →", 8.2, "start", MUTED)
    return svg(680, ly + 24, b)


# ---------------------------------------------------------------- figure 5: set selection vs top-k (schematic)
MOD = {"speech": ("#0f766e", "#e7f4f2", "speech segment"), "slide": ("#1d5fbf", "#e8f0fc", "slide text"),
       "figure": ("#7c3aed", "#f1eafd", "slide figure"), "notes": ("#b45309", "#fdf2e4", "notes paragraph")}


def fig5():
    def unit(x, y, kind, label, dim=False):
        st, fl, _ = MOD[kind]
        op = ' opacity="0.45"' if dim else ""
        glyph = {"speech": lambda: ic_wave(x + 5, y + 6, 14, 10, st, n=5), "slide": lambda: ic_slide(x + 5, y + 5, 15, 11, st),
                 "figure": lambda: ic_chart(x + 6, y + 5, 13, 11, st), "notes": lambda: ic_doc(x + 7, y + 4, 10, 14, st)}[kind]()
        return (f'<g{op}><rect x="{x}" y="{y}" width="150" height="22" rx="4" fill="{fl}" stroke="{st}" stroke-width="1.1"/>'
                + glyph + text(x + 26, y + 15, label, 8.6, fill=INK) + "</g>")
    b = text(0, 12, "SAME CANDIDATE POOL, TWO WAYS TO PICK THE EVIDENCE", 8.8, fill=MUTED, weight="700")
    left = [("speech", "speech 1: odds by father's age"), ("speech", "speech 2: same, reworded"), ("speech", "speech 3: same, reworded"),
            ("slide", "slide text: chart title"), ("speech", "speech 4: lead-in"), ("speech", "speech 5: same, reworded")]
    right = [("speech", "speech 1: odds by father's age"), ("slide", "slide text: chart title"), ("figure", "slide figure: study name"),
             ("notes", "notes: study design"), ("speech", "speech 4: lead-in"), ("slide", "slide text: next bullet")]
    x1, x2, y0, dy = 10, 360, 44, 30
    b += text(x1, 34, "(a) Plain top-k: each unit ranked alone", 9.6, weight="700")
    b += text(x2, 34, "(b) Complementarity set reranker", 9.6, weight="700")
    for i, (k, l) in enumerate(left):
        b += unit(x1, y0 + i * dy, k, l, dim=i in (1, 2, 5))
    for i, (k, l) in enumerate(right):
        b += unit(x2, y0 + i * dy, k, l)
    # annotations, left
    b += path(f"M {x1 + 158} {y0 + dy + 11} C {x1 + 190} {y0 + dy + 11}, {x1 + 190} {y0 + 2 * dy + 11}, {x1 + 160} {y0 + 2 * dy + 11}", GREY, True, "ag")
    b += text(x1 + 198, y0 + dy + 24, "near-duplicates fill", 8.4, fill=MUTED, style="italic")
    b += text(x1 + 198, y0 + dy + 35, "the set", 8.4, fill=MUTED, style="italic")
    b += text(x1 + 182, y0 + 5 * dy + 15, "figure never reached:", 8.4, fill=ORANGE, style="italic")
    b += text(x1 + 182, y0 + 5 * dy + 26, "it shares no words", 8.4, fill=ORANGE, style="italic")
    # annotations, right: same-slide link and the three terms
    yS, yF = y0 + dy + 11, y0 + 2 * dy + 11
    b += f'<path d="M {x2 + 150} {yS} C {x2 + 178} {yS}, {x2 + 178} {yF}, {x2 + 152} {yF}" fill="none" stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#ab)"/>'
    tx = x2 + 186
    b += text(tx, yS + 4, "+β  same-slide link", 8.6, fill=BLUE, weight="700")
    b += text(tx, yS + 15, "to a unit already chosen", 8.4, fill=MUTED)
    b += text(tx, y0 + 3 * dy + 10, "+α  a modality not yet", 8.6, fill=INK, weight="700")
    b += text(tx, y0 + 3 * dy + 21, "in the set", 8.4, fill=MUTED)
    b += text(tx, y0 + 5 * dy - 2, "−γ  charged for repeating", 8.6, fill=INK, weight="700")
    b += text(tx, y0 + 5 * dy + 9, "a unit of the same modality", 8.4, fill=MUTED)
    ly = y0 + 6 * dy + 14
    for i, (k, (st, fl, lab)) in enumerate(MOD.items()):
        x = 10 + i * 150
        b += f'<rect x="{x}" y="{ly}" width="22" height="12" rx="3" fill="{fl}" stroke="{st}" stroke-width="1.1"/>' + text(x + 28, ly + 10, lab, 8.6)
    b += text(676, ly + 10, "k = 6 shown", 8.4, "end", MUTED, style="italic")
    return svg(680, ly + 16, b)


# ---------------------------------------------------------------- figure 6: the four XFile conditions
def fig6():
    """The four XFile conditions, each with a schematic of the evidence set it hands the answerer."""
    L, T, cw, ch, gx, gy = 150, 46, 214, 86, 76, 50
    TEAL, PURP, AMB = "#0f766e", "#7c3aed", "#b45309"
    col = {"T": TEAL, "S": BLUE, "F": PURP, "N": AMB}
    strips = {(0, 0): "TTSTTTST", (0, 1): "TTSTTTST", (1, 0): "TSNTSTNT", (1, 1): "TSFNTSTN"}
    cells = {(0, 0): ("Plain top-k", "hybrid retrieval, top eight", "qry", False),
             (0, 1): ("Link expansion, top-k", "neighbours added; top eight unchanged", "link", True),
             (1, 0): ("No-links control", "reranker on the retrieved pool", "qry", False),
             (1, 1): ("LinkRAG", "links + reranker", "link", True)}
    pos = lambda r, c: (L + c * (cw + gx), T + r * (ch + gy))
    b = text(L + cw / 2, 30, "No links", 9.8, "middle", weight="700") + text(L + cw + gx + cw / 2, 30, "Links (one hop)", 9.8, "middle", weight="700")
    b += text(L - 14, T + ch / 2 - 2, "No reranker", 9.8, "end", weight="700") + text(L - 14, T + ch / 2 + 11, "(top eight)", 8.6, "end", MUTED)
    b += text(L - 14, T + ch + gy + ch / 2 - 2, "Complementarity", 9.8, "end", weight="700") + text(L - 14, T + ch + gy + ch / 2 + 11, "reranker (Eq. 7)", 8.6, "end", MUTED)
    for (r, c), (t, sub, st, pr) in cells.items():
        x, y = pos(r, c)
        b += card(x, y, cw, ch, t, [sub], st, None, pr)
        b += text(x + 9, y + 52, "evidence set (k = 8):", 7.8, fill=MUTED)
        for k, m in enumerate(strips[(r, c)]):
            b += f'<rect x="{x + 9 + k * 24}" y="{y + 58}" width="20" height="16" rx="3" fill="{col[m]}" fill-opacity="0.85"/>'
        if (r, c) == (1, 1):   # the same-slide link that brings the figure in
            x1, x2 = x + 9 + 1 * 24 + 10, x + 9 + 2 * 24 + 10
            b += f'<path d="M {x1} {y + 75} C {x1} {y + 84}, {x2} {y + 84}, {x2} {y + 76}" fill="none" stroke="{BLUE}" stroke-width="1.5" stroke-dasharray="3 2" marker-end="url(#ab)"/>'
    (ax, ay), (bx, by) = pos(1, 0), pos(1, 1)
    (px, py), (qx, qy) = pos(0, 0), pos(0, 1)
    b += line(ax + cw + 2, ay + ch / 2, bx - 3, by + ch / 2, BLUE, False, "ab", 2.2) + badge((ax + cw + bx) / 2, ay + ch / 2 - 13, "1", BLUE)
    b += line(px + cw + 2, py + ch / 2, qx - 3, qy + ch / 2, GREY, True, "ag") + badge((px + cw + qx) / 2, py + ch / 2 - 13, "3", GREY)
    b += path(f"M {px + cw - 20} {py + ch + 2} L {bx + 20} {by - 3}", INK, False, "ah", 1.4) + badge((px + cw + bx) / 2 + 2, (py + ch + by) / 2 - 2, "2", INK)
    b += line(px + 26, py + ch + 2, ax + 26, ay - 3, GREY, True, "ag") + badge(px + 12, (py + ch + ay) / 2, "4", GREY)
    ly = T + 2 * ch + gy + 22
    for i, (m, lab) in enumerate((("T", "speech"), ("S", "slide text"), ("F", "slide figure"), ("N", "notes"))):
        b += f'<rect x="{L + i * 96}" y="{ly - 9}" width="14" height="11" rx="2" fill="{col[m]}" fill-opacity="0.85"/>' + text(L + i * 96 + 19, ly, lab, 8.6)
    b += text(L + 4 * 96 + 6, ly, "(schematic)", 8.4, fill=MUTED, style="italic")
    notes = [("1", BLUE, "Headline, recorded before the test run: links, holding the reranker fixed."),
             ("2", INK, "Against plain top-k: links and reranker together, added after the tune check (§4.1)."),
             ("3", GREY, "Link expansion alone: cannot change the top eight (§3.2.3)."),
             ("4", GREY, "Reranker alone, on raw fusion scores: unstable (§5.3).")]
    for i, (n, c_, s_) in enumerate(notes):
        y = ly + 22 + i * 17
        b += badge(L - 112, y - 3, n, c_) + text(L - 98, y + 1, s_, 8.8)
    return svg(680, ly + 22 + 3 * 17 + 10, b)


# ---------------------------------------------------------------- figure 8: the F1 range on LectQA-Vid
def fig8():
    rows = [("Question echoed as the answer", 22.87, "ref"), ("Temporal-Aware RAG, published [2]", 23.52, "ref"),
            ("[2] re-implemented, filter on", 31.23, "base"), ("[2] re-implemented, filter off", 34.69, "base"),
            ("LinkRAG hybrid, no links", 34.96, "prop"), ("LinkRAG hybrid + re-query, no links", 36.70, "prop"),
            ("Full transcript, no retrieval", 38.74, "ceil")]
    col = {"ref": GREY, "base": "#7c8794", "prop": BLUE, "ceil": STAGE["ans"][0]}
    L, R, top, rh = 210, 640, 26, 24
    sx = lambda v: L + (R - L) * v / 45
    b = text((sx(22.87) + sx(38.74)) / 2, 12, "range a retrieval method can occupy", 8.4, "middle", MUTED, "700")
    b += text(sx(22.87), 12, "floor", 8.2, "end", MUTED, style="italic") + text(sx(38.74) + 3, 12, "ceiling", 8.2, "start", MUTED, style="italic")
    # usable range shading
    b += f'<rect x="{sx(22.87):.1f}" y="{top - 4}" width="{sx(38.74) - sx(22.87):.1f}" height="{rh * len(rows) + 4}" fill="#f4f6f9"/>'
    for t in range(0, 46, 5):
        b += f'<line x1="{sx(t):.1f}" y1="{top - 4}" x2="{sx(t):.1f}" y2="{top + rh * len(rows)}" stroke="{RULE}" stroke-width="0.5"/>' + text(sx(t), top + rh * len(rows) + 13, t, 8.4, "middle", MUTED)
    for i, (name, v, k) in enumerate(rows):
        y = top + i * rh
        b += text(L - 8, y + 14, name, 9, "end", style="italic" if k == "ref" else "normal", fill=MUTED if k == "ref" else INK, weight="700" if k == "prop" else "normal")
        b += f'<rect x="{L}" y="{y + 4}" width="{sx(v) - L:.1f}" height="14" rx="2" fill="{col[k]}"/>'
        b += text(sx(v) + 5, y + 15, f"{v:.2f}", 9, weight="700")
    b += text((L + R) / 2, top + rh * len(rows) + 27, "Overall token F1 (%), 300 open-ended questions, mean of three repeats", 8.6, "middle", MUTED)
    return svg(680, top + rh * len(rows) + 32, b)


# ---------------------------------------------------------------- figure 9: MaViLS per lecture
def fig9():
    rows = [("Phonetics", 0.99, 0.65), ("Solar resource", 0.81, 0.53), ("Climate &amp; Cities", 0.98, 0.86), ("Numerics", 0.89, 0.81),
            ("Computation theory", 0.87, 0.84), ("Deep learning", 0.99, 0.99), ("ML for health", 0.91, 0.95),
            ("Reinforcement", 0.67, 0.75), ("Short range", 0.58, 0.80)]
    L, R, top, rh = 140, 580, 34, 20
    sx = lambda v: L + (R - L) * (v - 0.4) / 0.6
    b = f'<circle cx="{L + 5}" cy="12" r="5" fill="{BLUE}"/>' + text(L + 14, 15.5, "LinkRAG alignment, with video-frame channels", 9)
    b += f'<rect x="{L + 250}" y="7.5" width="9" height="9" fill="#fff" stroke="{ORANGE}" stroke-width="2"/>' + text(L + 265, 15.5, "MaViLS, all features, published [3]", 9)
    n = len(rows) + 1
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += f'<line x1="{sx(t):.1f}" y1="{top - 4}" x2="{sx(t):.1f}" y2="{top + rh * n + 4}" stroke="{RULE}" stroke-width="0.5"/>' + text(sx(t), top + rh * n + 17, f"{t:.1f}", 8.4, "middle", MUTED)
    for i, (name, ours, theirs) in enumerate(rows + [("Mean of the nine", 0.853, 0.798)]):
        mean = i == len(rows)
        y = top + i * rh + rh / 2 + (6 if mean else 0)
        if mean:
            b += f'<line x1="0" y1="{y - 13}" x2="680" y2="{y - 13}" stroke="{RULE}" stroke-width="0.6"/>'
        b += text(L - 10, y + 3.2, name, 9, "end", weight="700" if mean else "normal")
        good = ours >= theirs
        b += f'<line x1="{sx(theirs):.1f}" y1="{y}" x2="{sx(ours):.1f}" y2="{y}" stroke="{BLUE if good else ORANGE}" stroke-width="2.2" opacity="0.55"/>'
        b += f'<rect x="{sx(theirs) - 4.5:.1f}" y="{y - 4.5}" width="9" height="9" fill="#fff" stroke="{ORANGE}" stroke-width="2"/>'
        b += f'<circle cx="{sx(ours):.1f}" cy="{y}" r="5" fill="{BLUE}" stroke="#fff" stroke-width="1.5"/>'
        d = ours - theirs
        lab = "tie" if abs(d) < 0.005 else (f"+{d:.3f}" if mean else f"{d:+.2f}".replace("-", "−"))
        b += text(R + 70, y + 3.2, lab, 9, "end", weight="700" if mean else "normal")
    b += text(R + 70, top - 8, "difference", 8.4, "end", MUTED)
    b += text((L + R) / 2, top + rh * n + 31, "F1 per lecture (authors' scorer)", 8.6, "middle", MUTED)
    return svg(680, top + rh * n + 36, b)


# ---------------------------------------------------------------- small drawn icons, shared by several figures
def ic_wave(x, y, w, h, col="#0f766e", hl=None, n=34):
    """Audio waveform; hl = (i0, i1) highlights bars i0..i1."""
    import math
    out, bw = "", w / n
    if hl:
        out += f'<rect x="{x + hl[0] * bw - 1:.1f}" y="{y - 2}" width="{(hl[1] - hl[0] + 1) * bw + 2:.1f}" height="{h + 4}" rx="3" fill="{col}" fill-opacity="0.16"/>'
    for i in range(n):
        a = 0.25 + 0.75 * abs(math.sin(i * 1.7) * math.cos(i * 0.45))
        bh = max(2, h * a)
        on = hl and hl[0] <= i <= hl[1]
        out += f'<rect x="{x + i * bw + bw * 0.2:.1f}" y="{y + (h - bh) / 2:.1f}" width="{bw * 0.6:.1f}" height="{bh:.1f}" rx="1" fill="{col}" fill-opacity="{1 if on or not hl else 0.45}"/>'
    return out


def ic_slide(x, y, w, h, col="#1d5fbf"):
    """A slide: frame, title bar, two bullet lines."""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="#fff" stroke="{col}" stroke-width="1.2"/>'
    out += f'<rect x="{x + w * 0.1:.1f}" y="{y + h * 0.14:.1f}" width="{w * 0.55:.1f}" height="{h * 0.11:.1f}" rx="1" fill="{col}"/>'
    for k in range(2):
        yy = y + h * (0.42 + k * 0.2)
        out += f'<circle cx="{x + w * 0.13:.1f}" cy="{yy:.1f}" r="{h * 0.04:.1f}" fill="{col}" fill-opacity="0.7"/>'
        out += f'<rect x="{x + w * 0.2:.1f}" y="{yy - h * 0.03:.1f}" width="{w * (0.6 - 0.15 * k):.1f}" height="{h * 0.06:.1f}" rx="1" fill="{col}" fill-opacity="0.45"/>'
    return out


def ic_doc(x, y, w, h, col="#b45309"):
    """A page with a folded corner and text lines."""
    f = w * 0.28
    out = (f'<path d="M {x} {y} H {x + w - f:.1f} L {x + w} {y + f:.1f} V {y + h} H {x} Z" fill="#fff" stroke="{col}" stroke-width="1.2"/>'
           f'<path d="M {x + w - f:.1f} {y} V {y + f:.1f} H {x + w}" fill="none" stroke="{col}" stroke-width="1"/>')
    for k in range(4):
        out += f'<rect x="{x + w * 0.15:.1f}" y="{y + h * (0.38 + k * 0.15):.1f}" width="{w * (0.7 if k % 3 else 0.5):.1f}" height="{h * 0.05:.1f}" rx="1" fill="{col}" fill-opacity="0.5"/>'
    return out


def ic_chart(x, y, w, h, col="#7c3aed", vals=(0.25, 0.3, 0.38, 0.5, 0.85)):
    out = f'<path d="M {x} {y} V {y + h} H {x + w}" fill="none" stroke="{col}" stroke-width="1"/>'
    bw = w / (len(vals) * 1.5)
    for i, v in enumerate(vals):
        out += f'<rect x="{x + bw * 0.5 + i * bw * 1.5:.1f}" y="{y + h - h * v:.1f}" width="{bw:.1f}" height="{h * v:.1f}" fill="{col}" fill-opacity="0.75"/>'
    return out


def ic_index(x, y, s, col="#0f766e"):
    """Stacked discs (an index)."""
    out = ""
    for k in range(3):
        yy = y + k * s * 0.28
        out += (f'<ellipse cx="{x + s / 2}" cy="{yy + s * 0.12:.1f}" rx="{s / 2}" ry="{s * 0.12:.1f}" fill="#fff" stroke="{col}" stroke-width="1.1"/>')
    out += f'<path d="M {x} {y + s * 0.12:.1f} V {y + s * 0.68:.1f} M {x + s} {y + s * 0.12:.1f} V {y + s * 0.68:.1f}" stroke="{col}" stroke-width="1.1"/>'
    return out


def ic_graph(x, y, s, col="#1d5fbf"):
    """A small node-link graph."""
    pts = [(0.15, 0.3), (0.5, 0.12), (0.85, 0.35), (0.3, 0.8), (0.75, 0.82)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (2, 4), (1, 4)]
    P = [(x + px * s, y + py * s) for px, py in pts]
    out = "".join(f'<line x1="{P[a][0]:.1f}" y1="{P[a][1]:.1f}" x2="{P[b][0]:.1f}" y2="{P[b][1]:.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="3 2"/>' for a, b in edges)
    cols = ["#0f766e", "#1d5fbf", "#7c3aed", "#b45309", "#1d5fbf"]
    out += "".join(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{s * 0.09:.1f}" fill="{c}"/>' for (px, py), c in zip(P, cols))
    return out


def ic_check(x, y, r, ok=True):
    col = "#15803d" if ok else "#b91c1c"
    mark = (f'<path d="M {x - r * 0.45:.1f} {y} L {x - r * 0.1:.1f} {y + r * 0.4:.1f} L {x + r * 0.5:.1f} {y - r * 0.4:.1f}" fill="none" stroke="#fff" stroke-width="{r * 0.3:.1f}" stroke-linecap="round"/>'
            if ok else f'<path d="M {x - r * 0.4:.1f} {y - r * 0.4:.1f} L {x + r * 0.4:.1f} {y + r * 0.4:.1f} M {x + r * 0.4:.1f} {y - r * 0.4:.1f} L {x - r * 0.4:.1f} {y + r * 0.4:.1f}" stroke="#fff" stroke-width="{r * 0.28:.1f}" stroke-linecap="round"/>')
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}"/>' + mark


def chip(x, y, label, col, fill):
    w = 7 + len(label) * 5.0
    return (f'<rect x="{x}" y="{y}" width="{w:.1f}" height="15" rx="7.5" fill="{fill}" stroke="{col}" stroke-width="1"/>'
            + text(x + w / 2, y + 10.8, label, 8, "middle", col, "700")), w


# right-aligned icons for the pipeline cards (x = right edge, y = top)
TOP_ICON = [lambda x, y: ic_wave(x - 54, y + 2, 16, 10, n=6) + ic_slide(x - 34, y, 15, 11) + ic_doc(x - 14, y - 1, 11, 14),
            lambda x, y: "".join(f'<rect x="{x - 40 + k * 13}" y="{y + 1}" width="10" height="10" rx="2" fill="{c}"/>' for k, c in enumerate(("#0f766e", "#1d5fbf", "#7c3aed"))),
            lambda x, y: ic_index(x - 22, y, 20),
            lambda x, y: ic_graph(x - 38, y - 2, 36)]
# centred icons under the bottom cards' text (x = centre, y = top)
BOT_ICON = [lambda x, y: f'<circle cx="{x - 4}" cy="{y + 10}" r="8" fill="none" stroke="#4b5563" stroke-width="2"/><line x1="{x + 2}" y1="{y + 16}" x2="{x + 10}" y2="{y + 24}" stroke="#4b5563" stroke-width="2.4" stroke-linecap="round"/>',
            lambda x, y: ic_graph(x - 15, y, 28),
            lambda x, y: "".join(f'<rect x="{x - 34 + k * 14}" y="{y + 8}" width="11" height="11" rx="2" fill="{c}"/>' for k, c in enumerate(("#0f766e", "#1d5fbf", "#7c3aed", "#b45309", "#0f766e"))),
            lambda x, y: f'<path d="M {x - 22} {y + 2} H {x + 22} V {y + 20} H {x - 6} L {x - 14} {y + 27} V {y + 20} H {x - 22} Z" fill="#fff" stroke="#b45309" stroke-width="1.3"/>' + "".join(f'<rect x="{x - 17}" y="{y + 7 + k * 6}" width="{30 - k * 10}" height="2.6" rx="1" fill="#b45309" fill-opacity="0.6"/>' for k in range(2)) + f'<rect x="{x + 6}" y="{y + 12}" width="10" height="5" rx="2.5" fill="#1d5fbf"/>',
            lambda x, y: ic_check(x, y + 13, 11)]

# ---------------------------------------------------------------- teaser: one question, three files
def fig_teaser():
    TEAL, PURP, AMB = "#0f766e", "#7c3aed", "#b45309"
    b = ""
    # the question
    b += f'<rect x="0" y="0" width="680" height="44" rx="10" fill="#f6f7f9" stroke="{RULE}" stroke-width="1"/>'
    b += f'<circle cx="22" cy="22" r="12" fill="{INK}"/>' + text(22, 27, "?", 13, "middle", "#fff", "700")
    b += text(44, 18, "“When the lecturer says ‘you can see’ how a father's age relates to schizophrenia, what odds does he give for", 9.4)
    b += text(44, 32, "fathers under 25 versus in their 50s, and what study does the graph on the slide come from?”", 9.4)
    # three panels
    py, ph = 62, 178
    panels = [(0, 214, "LECTURE RECORDING", TEAL), (233, 214, "SLIDE DECK", BLUE), (466, 214, "TEXTBOOK CHAPTER", AMB)]
    for x, w, t, c in panels:
        b += f'<rect x="{x}" y="{py}" width="{w}" height="{ph}" rx="8" fill="#fff" stroke="{c}" stroke-width="1.3"/>'
        b += f'<rect x="{x}" y="{py}" width="{w}" height="22" rx="8" fill="{c}"/><rect x="{x}" y="{py + 12}" width="{w}" height="10" fill="{c}"/>'
        b += text(x + 10, py + 15, t, 8.8, fill="#fff", weight="700")
    # recording: waveform with the spoken segment highlighted
    b += ic_wave(12, py + 34, 190, 30, TEAL, hl=(14, 22))
    b += text(12, py + 78, "speech segment, 18:39", 8.4, fill=TEAL, weight="700")
    for i, l in enumerate(["“…And you can see if the father is", "less than 25, it's 1 out of 141. If", "the father is in his 50s, it becomes", "1 out of 50…”"]):
        b += text(12, py + 94 + i * 12.5, l, 8.6, fill=INK, style="italic")
    b += text(12, py + ph - 10, "has the odds · never names the study", 8.2, fill=MUTED)
    # slide 17: text block + figure
    sx0 = 245
    b += f'<rect x="{sx0}" y="{py + 30}" width="190" height="108" rx="3" fill="#fbfcfe" stroke="{BLUE}" stroke-width="1"/>'
    b += text(sx0 + 8, py + 44, "Slide 17", 8.4, fill=BLUE, weight="700")
    b += f'<rect x="{sx0 + 6}" y="{py + 50}" width="178" height="26" rx="2" fill="{BLUE}" fill-opacity="0.08" stroke="{BLUE}" stroke-width="0.8" stroke-dasharray="3 2"/>'
    b += text(sx0 + 10, py + 61, "text: “Estimated cumulative incidence…", 7.8, fill=INK)
    b += text(sx0 + 10, py + 71, "onset of schizophrenia by age 34”", 7.8, fill=INK)
    b += f'<rect x="{sx0 + 6}" y="{py + 82}" width="178" height="52" rx="2" fill="{PURP}" fill-opacity="0.07" stroke="{PURP}" stroke-width="0.8" stroke-dasharray="3 2"/>'
    b += ic_chart(sx0 + 14, py + 88, 70, 30, PURP)
    b += text(sx0 + 14, py + 128, "&lt;25 … 40–44", 7, fill=MUTED)
    b += text(sx0 + 92, py + 98, "figure: “Data from", 7.8, fill=INK)
    b += text(sx0 + 92, py + 109, "Malaspina et al.,", 7.8, fill=PURP, weight="700")
    b += text(sx0 + 92, py + 120, "Arch Gen Psychiatry", 7.8, fill=PURP, weight="700")
    b += text(sx0 + 92, py + 131, "(2001)”", 7.8, fill=PURP, weight="700")
    b += text(245, py + ph - 10, "only the figure names the study", 8.2, fill=MUTED)
    # links: speech -> slide text (audio-slide), slide text -> figure (same-slide)
    b += path(f"M 205 {py + 49} C 228 {py + 49}, 228 {py + 63}, {sx0 + 4} {py + 63}", BLUE, True, "ab", 1.6)
    b += text(202, py + 78, "audio–slide link →", 7.8, "end", BLUE, "700")
    b += path(f"M {sx0 + 186} {py + 66} C {sx0 + 200} {py + 70}, {sx0 + 200} {py + 104}, {sx0 + 186} {py + 108}", BLUE, True, "ab", 1.6)
    b += text(sx0 + 184, py + 150, "same-slide link", 7.8, "end", BLUE, "700")
    # textbook: present in the index, not needed here
    b += ic_doc(478, py + 34, 46, 58, AMB)
    for i, l in enumerate(["OpenStax Psychology 2e,", "ch. 2, research methods"]):
        b += text(532, py + 50 + i * 12, l, 8.4, fill=INK)
    for i, l in enumerate(["In the same pooled index.", "Not needed for this question;", "other XFile questions need", "a fact from it."]):
        b += text(478, py + 112 + i * 12.5, l, 8.4, fill=MUTED)
    # outcomes
    oy = py + ph + 18
    b += text(0, oy, "WHAT EACH SYSTEM PUTS IN FRONT OF THE ANSWERER", 8.8, fill=MUTED, weight="700")
    rows = [("Plain top-k", [("speech", TEAL, "#e7f4f2"), ("slide text", BLUE, "#e8f0fc")], False,
             "odds right, study missing: the figure shares no words with the question"),
            ("LinkRAG", [("speech", TEAL, "#e7f4f2"), ("slide text", BLUE, "#e8f0fc"), ("+ slide figure", PURP, "#f1eafd")], True,
             "same-slide link pulls in the figure: odds and study both cited")]
    for i, (name, chips, ok, note) in enumerate(rows):
        y = oy + 12 + i * 26
        b += text(0, y + 11, name, 9.4, weight="700")
        x = 80
        for lab, c, f in chips:
            g, w = chip(x, y, lab, c, f); b += g; x += w + 6
        b += ic_check(x + 10, y + 7.5, 7.5, ok) + text(x + 24, y + 11, note, 8.6, fill=INK if ok else MUTED)
    return svg(680, oy + 12 + 2 * 26 + 2, b)


CAPTIONS = {
    "fig1": "<b>Figure 1.</b> Construction of XFile. (a) Each stage removes or relabels questions: ① drafting, ② human review, ③ the three-run gold check, which keeps a cross-file label only when every single-source answer fails and the all-source answer passes, ④ the verified cross-file subset. (b) The verified questions by tune/test split, fixed before any evaluation run.",
    "fig2": "<b>Figure 2.</b> The proposed LinkRAG pipeline. Steps ①–④ run once per corpus; steps ⑤–⑨ run for every question. Dashed blue boxes are the proposed components; the blue dashed arrows show the links built in ④ being used in ⑥ and ⑦.",
    "fig3": "<b>Figure 3.</b> Links built by the Evidence Linking Layer on XFile, with their scores. ①–③: a chain from a sentence of the reinforcement-learning lecture to its slide, the figure on that slide and a textbook paragraph. ④: a deictic link from the cryptocurrency lecture, the rejected Lightning topic, which holds every deictic link on a correct recording–deck pair (§5.4). All units, links and scores are taken unchanged from the built link graph.",
    "fig4": "<b>Figure 4.</b> The relatedness gate on the pooled XFile corpus, all 49 recording × deck pairs. Pairs to the right of the threshold receive audio–slide links. Five recordings are linked to their own deck; two own-deck pairs are rejected and four pairs of different lectures are admitted (§5.4).",
    "fig5": "<b>Figure 5.</b> How the set reranker of Eqs. 7–8 differs from top-k ranking, shown schematically on a question like the one in §1.1 (illustrative units, not a measured case). (a) Ranked one at a time, near-paraphrases of the same spoken sentence fill the set and the slide figure, which shares no words with the question, is never reached. (b) The set objective rewards a modality not yet covered (α) and a link to a unit already chosen (β), and charges for same-modality repetition (γ).",
    "fig6": "<b>Figure 6.</b> The four XFile conditions: links (columns) crossed with the reranker (rows). Each arrow is one comparison and names what it isolates. Each cell also sketches the evidence set it hands the answerer (schematic, coloured by modality); only LinkRAG can follow the same-slide link to the figure. Arrow 1 is the headline recorded before the test run; 2 was added after the tune check; 3 is empty by construction; 4 is affected by the score-scale issue of §5.3. Results are in Table 3.",
    "fig7": "<b>Figure 7.</b> Paired differences in answer correctness on the XFile test half (three topics, 38 questions): LinkRAG against the pre-registered no-links control (squares) and against plain top-k retrieval (circles), with 95 % intervals from resampling topics. With three topics each interval is the range of the per-topic differences, so it cannot establish a difference; against plain top-k the two systems answer identically.",
    "fig8": "<b>Figure 8.</b> Overall token F1 on LectQA-Vid open-ended questions. Light grey: references that need no retrieval system; dark grey: the re-implementation of [2]; blue: LinkRAG hybrid retrieval, without links; amber: the whole transcript, no retrieval. The shaded band is the usable range between the question-echo floor and the full-transcript ceiling.",
    "fig9": "<b>Figure 9.</b> MaViLS, the nine held-out lectures with video, ordered by difference. Gains are on lectures where the published system scores low; losses are on camera-heavy lectures. The mean difference is +0.055 with 95 % CI [−0.050, +0.161].",
}
TEASER_CAPTION = ("One XFile question whose answer needs two files (PS09, a tune question; real units from the "
                  "corpus). The odds are spoken in the recording; the study they come from is printed only in the slide's figure, "
                  "which shares no words with the question. Plain top-k finds the speech and the slide's text but not the figure; "
                  "the same-slide link brings the figure in, and LinkRAG answers in all three runs. This shows the design intent, "
                  "not the result: on the held-out test topics LinkRAG and plain top-k answered every question identically (§4.1).")
# figure order in the paper; numbers and ids (fig1, fig2, ...) follow from position
ORDER = [(fig_teaser, TEASER_CAPTION)] + [(f, re.sub(r"^<b>Figure \d+\.</b> ", "", CAPTIONS[k]))
                                          for k, f in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5),
                                                       ("fig6", fig6), ("fig7", fig7), ("fig8", fig8), ("fig9", fig9))]

if __name__ == "__main__":
    import os
    p = os.environ.get("PAPER_HTML") or "paper/paper.html"
    t = open(p).read()
    have_results = os.path.exists(os.environ.get("XFILE_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "xfile_results.json"))
    for n, (fn, cap) in enumerate(ORDER, 1):
        k = f"fig{n}"
        if fn is fig7 and not have_results:
            cap = '<span class="todo">[[XFILE: caption pending results]]</span>'
        pat = re.compile(rf'<figure id="{k}">.*?</figure>', re.S)
        assert len(pat.findall(t)) == 1, f"{k}: expected exactly one placeholder"
        t = pat.sub(lambda _m: f'<figure id="{k}">{fn()}<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>', t)
    open(p, "w").write(t)
    print("figures inserted:", t.count("<figure"))
