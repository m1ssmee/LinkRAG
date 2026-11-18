"""Build the paper's inline SVG figures and insert them into paper/paper.html (idempotent).

    python paper/figures.py && python scripts/render_pdf.py paper/paper.html paper/LinkRAG_paper.pdf

paper.html holds one <figure id="figN">…</figure> per figure; this script replaces each one whole.
Numbers are copied from committed sources:
  fig1  tests/regression/xfile_questions.jsonl, results/xfile/gold_check_3runs.md, reports/xfile_review.csv
  fig3  data/processed/xfile/links.jsonl (unit ids and scores also in results/xfile/links.jsonl)
  fig4  build_links log, audio_slide gate lines (DESIGN.md known issue (n))
  fig5  results/xfile/xfile_benchmark.md (pending)
  fig6  results/external/lectqa_recheck.md
  fig7  results/external/mavils_recheck.md
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
        if i:
            px, pw = top[i - 1][0], top[i - 1][1]
            b += line(px + pw + 1, y1 + h / 2, x - 2, y1 + h / 2)
    y2 = 168
    b += text(0, y2 - 10, "QUESTION TIME", 8.8, fill=MUTED, weight="700")
    b += f'<rect x="0" y="{y2 + 22}" width="62" height="26" rx="13" fill="#fff" stroke="{INK}" stroke-width="1.1"/>' + text(31, y2 + 39, "Question", 9.2, "middle", weight="700")
    bot = [(82, "5", "Hybrid retrieval", ["dense + BM25,", "rank fusion"], "qry", False),
           (204, "6", "Link expansion", ["add 1-hop linked", "neighbours of seeds"], "link", True),
           (326, "7", "Set reranker", ["coverage + links", "− redundancy"], "link", True),
           (448, "8", "Answerer", ["claims, each citing", "evidence units"], "ans", False),
           (570, "9", "Verifier", ["each claim checked", "against its citation"], "ans", False)]
    w2, h2 = 110, 70
    b += line(62, y2 + 35, 80, y2 + 35)
    for i, (x, n, t, ls, st, pr) in enumerate(bot):
        b += card(x, y2, w2, h2, t, ls, st, n, pr)
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
    g1, g2 = (C1 + w + C2) / 2, (C2 + w + C3) / 2
    b += card(C1, 22, w, 82, "Speech, 06:40", ["“…a sequence of random", "states S1, S2, … with the Markov", "property. That's the definition", "of a Markov process…”"], "qry", size=8.4)
    b += card(C2, 22, w, 82, "Slide 6, text block", ["“Markov Process: a memoryless", "random process, i.e. a sequence", "of random states S1, S2, … with", "the Markov property.”"], "qry", size=8.4)
    b += card(C2, 140, w, 64, "Slide 6, figure", ["definition box: tuple ⟨S, P⟩", "and transition matrix P(ss′)"], "qry", size=8.4)
    b += card(C3, 140, w, 64, "Ch. 3 paragraph", ["recycling-robot example:", "transition probabilities α, β"], "qry", size=8.4)
    b += line(C1 + w + 2, 63, C2 - 2, 63, BLUE, marker="ab", width=1.8) + badge(g1, 47, "1", BLUE)
    b += text(g1, 80, "audio–slide", 8.4, "middle", BLUE, "700") + text(g1, 91, "0.81", 8.4, "middle", BLUE)
    b += line(C2 + w / 2, 138, C2 + w / 2, 106, BLUE, marker="ab", width=1.8) + badge(C2 + w / 2 - 16, 122, "2", BLUE)
    b += text(C2 + w / 2 + 8, 125, "same-slide 1.00", 8.4, fill=BLUE, weight="700")
    b += line(C2 + w + 2, 172, C3 - 2, 172, BLUE, marker="ab", width=1.8) + badge(g2, 156, "3", BLUE)
    b += text(g2, 189, "figure–text", 8.4, "middle", BLUE, "700") + text(g2, 200, "0.60", 8.4, "middle", BLUE)
    y = 226
    b += f'<line x1="0" y1="{y - 8}" x2="680" y2="{y - 8}" stroke="{RULE}" stroke-width="0.8"/>'
    b += card(C1, y, w, 64, "Speech, 01:26 (other lecture)", ["“…now Bob can use Bob's key", "and this revocable key…”"], "qry", size=8.4)
    b += card(C2, y, w, 64, "Slide 4, figure", ["diagram labelled", "“Commit Tx (held by Bob)”"], "qry", size=8.4)
    b += line(C1 + w + 2, y + 32, C2 - 2, y + 32, BLUE, marker="ab", width=1.8) + badge(g1, y + 16, "4", BLUE)
    b += text(g1, y + 49, "deictic", 8.4, "middle", BLUE, "700") + text(g1, y + 60, "0.65", 8.4, "middle", BLUE)
    b += text(C3, y + 22, "A pointing phrase (“this …”) is", 8.6, fill=MUTED)
    b += text(C3, y + 35, "linked to the visual element on", 8.6, fill=MUTED)
    b += text(C3, y + 48, "the slide on screen at that moment.", 8.6, fill=MUTED)
    return svg(680, y + 70, b)


# ---------------------------------------------------------------- figure 4: the relatedness gate on XFile
GATE = [("psych_methods","rl_mdp",-0.6), ("psych_methods","gp_scaling",-0.6), ("psych_methods","nfa_closure",-0.7), ("psych_methods","mimic_data",-0.1), ("psych_methods","psych_methods",12.2), ("psych_methods","visual_system",1.4), ("psych_methods","lightning",-0.2), ("visual_system","rl_mdp",-0.3), ("visual_system","gp_scaling",-0.4), ("visual_system","nfa_closure",-0.2), ("visual_system","mimic_data",-0.2), ("visual_system","psych_methods",-0.2), ("visual_system","visual_system",-0.1), ("visual_system","lightning",-0.1), ("lightning","rl_mdp",0.0), ("lightning","gp_scaling",-0.8), ("lightning","nfa_closure",-0.7), ("lightning","mimic_data",-1.0), ("lightning","psych_methods",2.6), ("lightning","visual_system",0.7), ("lightning","lightning",4.9), ("rl_mdp","rl_mdp",3.9), ("rl_mdp","gp_scaling",-1.6), ("rl_mdp","nfa_closure",1.4), ("rl_mdp","mimic_data",-0.0), ("rl_mdp","psych_methods",-0.2), ("rl_mdp","visual_system",-0.0), ("rl_mdp","lightning",-0.3), ("gp_scaling","rl_mdp",-0.2), ("gp_scaling","gp_scaling",8.0), ("gp_scaling","nfa_closure",2.1), ("gp_scaling","mimic_data",-0.2), ("gp_scaling","psych_methods",0.2), ("gp_scaling","visual_system",-0.0), ("gp_scaling","lightning",-0.5), ("mimic_data","rl_mdp",-0.3), ("mimic_data","gp_scaling",-0.4), ("mimic_data","nfa_closure",-0.4), ("mimic_data","mimic_data",-1.5), ("mimic_data","psych_methods",-0.1), ("mimic_data","visual_system",0.2), ("mimic_data","lightning",0.3), ("nfa_closure","rl_mdp",-0.3), ("nfa_closure","gp_scaling",-0.7), ("nfa_closure","nfa_closure",2.3), ("nfa_closure","mimic_data",-0.0), ("nfa_closure","psych_methods",0.5), ("nfa_closure","visual_system",0.3), ("nfa_closure","lightning",0.7)]
SHORT = {"psych_methods": "psych", "visual_system": "visual", "lightning": "lightning", "rl_mdp": "rl", "gp_scaling": "gp", "mimic_data": "mimic", "nfa_closure": "nfa"}


def fig4():
    Z = 1.27
    L, R = 170, 660
    lo, hi = -2.0, 13.0
    sx = lambda z: L + (R - L) * (z - lo) / (hi - lo)
    rows = [("Recording × its own deck", 7, 60, [g for g in GATE if g[0] == g[1]]),
            ("Recording × another lecture's deck", 42, 140, [g for g in GATE if g[0] != g[1]])]
    b = ""
    axis_y = 190
    for t in range(-2, 14, 2):
        b += f'<line x1="{sx(t):.1f}" y1="22" x2="{sx(t):.1f}" y2="{axis_y}" stroke="{RULE}" stroke-width="0.5"/>' + text(sx(t), axis_y + 13, t, 8.4, "middle", MUTED)
    b += text((L + R) / 2, axis_y + 27, "z: path score of the alignment against 30 shuffled slide orders (shuffled std devs)", 8.6, "middle", MUTED)
    b += f'<rect x="{sx(Z):.1f}" y="18" width="{R - sx(Z):.1f}" height="{axis_y - 18}" fill="{STAGE["link"][1]}" opacity="0.55"/>'
    b += f'<line x1="{sx(Z):.1f}" y1="14" x2="{sx(Z):.1f}" y2="{axis_y}" stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="5 3"/>'
    b += text(sx(Z) + 5, 14, "z = 1.27: links admitted to the right", 8.6, fill=BLUE, weight="700")
    for name, n, yc, pts in rows:
        b += text(L - 10, yc + 3, name, 9, "end", weight="700") + text(L - 10, yc + 15, f"{n} pairs", 8.4, "end", MUTED)
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
    items = [("full", BLUE, "own deck, admitted (5)"), ("hollow", ORANGE, "own deck, rejected (2)"),
             ("full", ORANGE, "other deck, admitted (4)"), ("hollow", GREY, "other deck, rejected (38)")]
    for i, (kind, c, lab) in enumerate(items):
        x = L - 160 + i * 168
        b += (f'<circle cx="{x + 5}" cy="{ly}" r="4.4" fill="{c}"/>' if kind == "full"
              else f'<circle cx="{x + 5}" cy="{ly}" r="4.2" fill="#fff" stroke="{c}" stroke-width="1.6"/>')
        b += text(x + 14, ly + 3.4, lab, 8.6)
    return svg(680, ly + 10, b)


# ---------------------------------------------------------------- figure 5: XFile results (pending)
def fig5():
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


# ---------------------------------------------------------------- figure 6: the F1 range on LectQA-Vid
def fig6():
    rows = [("Question echoed as the answer", 22.87, "ref"), ("Temporal-Aware RAG, published [2]", 23.52, "ref"),
            ("[2] re-implemented, filter on", 31.23, "base"), ("[2] re-implemented, filter off", 34.69, "base"),
            ("LinkRAG, hybrid retrieval", 34.96, "prop"), ("LinkRAG, hybrid + one re-query", 36.70, "prop"),
            ("Full transcript, no retrieval", 38.74, "ceil")]
    col = {"ref": GREY, "base": "#7c8794", "prop": BLUE, "ceil": STAGE["ans"][0]}
    L, R, top, rh = 210, 640, 10, 24
    sx = lambda v: L + (R - L) * v / 45
    b = ""
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


# ---------------------------------------------------------------- figure 7: MaViLS per lecture
def fig7():
    rows = [("Phonetics", 0.99, 0.65), ("Solar resource", 0.81, 0.53), ("Climate &amp; Cities", 0.98, 0.86), ("Numerics", 0.89, 0.81),
            ("Computation theory", 0.87, 0.84), ("Deep learning", 0.99, 0.99), ("ML for health", 0.91, 0.95),
            ("Reinforcement", 0.67, 0.75), ("Short range", 0.58, 0.80)]
    L, R, top, rh = 140, 580, 34, 20
    sx = lambda v: L + (R - L) * (v - 0.4) / 0.6
    b = f'<circle cx="{L + 5}" cy="12" r="5" fill="{BLUE}"/>' + text(L + 14, 15.5, "LinkRAG audio–slide alignment", 9)
    b += f'<rect x="{L + 200}" y="7.5" width="9" height="9" fill="#fff" stroke="{ORANGE}" stroke-width="2"/>' + text(L + 215, 15.5, "MaViLS, all features, published [3]", 9)
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


CAPTIONS = {
    "fig1": "<b>Figure 1.</b> Construction of XFile. (a) Each stage removes or relabels questions: ① drafting, ② human review, ③ the three-run gold check, which keeps a cross-file label only when every single-source answer fails and the all-source answer passes, ④ the verified cross-file subset. (b) The verified questions by tune/test split, fixed before any evaluation run.",
    "fig2": "<b>Figure 2.</b> The proposed LinkRAG pipeline. Steps ①–④ run once per corpus; steps ⑤–⑨ run for every question. Dashed blue boxes are the proposed components; the blue dashed arrows show the links built in ④ being used in ⑥ and ⑦.",
    "fig3": "<b>Figure 3.</b> Links built by the Evidence Linking Layer on XFile, with their scores. ①–③: a chain from a sentence of the reinforcement-learning lecture to its slide, the figure on that slide and a textbook paragraph. ④: a deictic link from the cryptocurrency lecture. All units, links and scores are taken unchanged from the built link graph.",
    "fig4": "<b>Figure 4.</b> The relatedness gate on the pooled XFile corpus, all 49 recording × deck pairs. Pairs to the right of the threshold receive audio–slide links. Five recordings are linked to their own deck; two own-deck pairs are rejected and four pairs of different lectures are admitted (§5.4).",
    "fig5": "<b>Figure 5.</b> Paired differences in answer correctness on the XFile test half (three topics, 38 questions): LinkRAG against the pre-registered no-links control (squares) and against plain top-k retrieval (circles), with 95 % intervals from resampling topics. Every interval contains zero; against plain top-k the two systems answer identically.",
    "fig6": "<b>Figure 6.</b> Overall token F1 on LectQA-Vid open-ended questions. Light grey: references that need no retrieval system; dark grey: the re-implementation of [2]; blue: LinkRAG; amber: the whole transcript, no retrieval. The shaded band is the usable range between the question-echo floor and the full-transcript ceiling.",
    "fig7": "<b>Figure 7.</b> MaViLS, the nine held-out lectures with video, ordered by difference. Gains are on lectures where the published system scores low; losses are on camera-heavy lectures. The mean difference is +0.055 with 95 % CI [−0.050, +0.161].",
}
BUILD = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7}

if __name__ == "__main__":
    p = "paper/paper.html"
    t = open(p).read()
    import os
    if not os.path.exists(os.environ.get("XFILE_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "xfile_results.json")):
        CAPTIONS["fig5"] = "<b>Figure 5.</b> <span class=\"todo\">[[XFILE: caption pending results]]</span>"
    if os.environ.get("PAPER_HTML"):
        p = os.environ["PAPER_HTML"]
        t = open(p).read()
    for k, fn in BUILD.items():
        pat = re.compile(rf'<figure id="{k}">.*?</figure>', re.S)
        assert len(pat.findall(t)) == 1, f"{k}: expected exactly one placeholder"
        t = pat.sub(lambda _m: f'<figure id="{k}">{fn()}<figcaption>{CAPTIONS[k]}</figcaption></figure>', t)
    open(p, "w").write(t)
    print("figures inserted:", t.count("<figure"))
