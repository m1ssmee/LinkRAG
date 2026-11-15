"""Build the paper's three inline SVG figures and insert them into paper/paper.html (idempotent).

    python paper/figures.py && python scripts/render_pdf.py paper/paper.html paper/LinkRAG_paper_draft.pdf

The numbers are copied from results/external/lectqa_recheck.md and mavils_recheck.md.
"""
import re
INK, MUTED, RULE, BLUE, ORANGE, GREY = "#111", "#555", "#bbb", "#2a78d6", "#eb6834", "#9a9a9a"
F = 'font-family="Helvetica, Arial, sans-serif"'

def svg(w, h, body):
    return f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">{body}</svg>'

def text(x, y, s, size=9.5, anchor="start", fill=INK, weight="normal", style="normal"):
    return f'<text x="{x}" y="{y}" {F} font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" font-style="{style}">{s}</text>'

# ---------------------------------------------------------------- figure 1: pipeline
def box(x, y, w, h, lines, dashed=False):
    d = ' stroke-dasharray="4 3"' if dashed else ""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{"#f3f6fb" if dashed else "#fff"}" stroke="{BLUE if dashed else INK}" stroke-width="1"{d}/>'
    y0 = y + h / 2 - (len(lines) - 1) * 5.5 + 3.2
    for i, l in enumerate(lines):
        out += text(x + w / 2, y0 + i * 11, l, 8.6, "middle", weight="600" if i == 0 else "normal", fill=INK if i == 0 else MUTED)
    return out

def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="1" marker-end="url(#a)"/>'

def fig1():
    b = '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#111"/></marker></defs>'
    b += text(0, 12, "Indexing time", 9, weight="600", fill=MUTED)
    top = [("Sources", "recording, slides,", "documents"), ("Evidence units", "modality + page", "or time span"), ("Index", "BGE-M3 dense", "+ BM25"), ("Linking layer", "audio–slide, figure–text,", "deictic, same-slide")]
    xs, w, h, y = [0, 165, 330, 495], 145, 46, 20
    for i, t in enumerate(top):
        b += box(xs[i], y, w, h, t, dashed=(i == 3))
        if i: b += arrow(xs[i - 1] + w, y + h / 2, xs[i] - 1, y + h / 2)
    y2 = 118
    b += text(0, y2 - 8, "Question time", 9, weight="600", fill=MUTED)
    bot = [("Question",), ("Hybrid retrieval", "reciprocal rank", "fusion"), ("Link expansion", "one hop", "along links"), ("Set reranker", "coverage,", "links, redundancy"), ("Answerer", "claims with", "citations"), ("Verifier", "each claim vs", "cited text")]
    ws = [62, 98, 92, 104, 86, 88]; gap = 22; x = 0; pos = []
    for i, t in enumerate(bot):
        b += box(x, y2, ws[i], h, t, dashed=i in (2, 3)); pos.append(x)
        if i: b += arrow(pos[i - 1] + ws[i - 1], y2 + h / 2, x - 1, y2 + h / 2)
        x += ws[i] + gap
    # index and links feed retrieval; the re-query loop
    b += f'<path d="M {xs[2] + w/2} {y + h} L {xs[2] + w/2} {y2 - 22} L {pos[1] + ws[1]/2} {y2 - 22} L {pos[1] + ws[1]/2} {y2 - 1}" fill="none" stroke="{INK}" stroke-width="1" marker-end="url(#a)"/>'
    b += f'<path d="M {xs[3] + w/2} {y + h} L {xs[3] + w/2} {y2 - 14} L {pos[2] + ws[2]/2} {y2 - 14} L {pos[2] + ws[2]/2} {y2 - 1}" fill="none" stroke="{BLUE}" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#a)"/>'
    ax, rx = pos[4] + ws[4] / 2, pos[1] + ws[1] / 2
    b += f'<path d="M {ax} {y2 + h} L {ax} {y2 + h + 18} L {rx} {y2 + h + 18} L {rx} {y2 + h + 1}" fill="none" stroke="{INK}" stroke-width="1" marker-end="url(#a)"/>'
    b += text((ax + rx) / 2, y2 + h + 30, "one follow-up query written by the answering model", 8.4, "middle", MUTED)
    return svg(640, 200, b)

# ---------------------------------------------------------------- figure 2: the F1 range on LectQA-Vid
def fig2():
    rows = [("Question echoed as the answer", 22.87, GREY), ("Temporal-Aware RAG, published [2]", 23.52, GREY),
            ("[2] re-implemented, filter on", 31.23, BLUE), ("[2] re-implemented, filter off", 34.69, BLUE),
            ("Ours: hybrid retrieval", 34.96, BLUE), ("Ours: hybrid + one re-query", 36.70, BLUE),
            ("Full transcript, no retrieval", 38.74, BLUE)]
    L, R, top, rh = 205, 600, 22, 22
    sx = lambda v: L + (R - L) * v / 45
    b = ""
    for t in range(0, 46, 5):
        b += f'<line x1="{sx(t)}" y1="{top - 4}" x2="{sx(t)}" y2="{top + rh * len(rows)}" stroke="{RULE}" stroke-width="0.5"/>' + text(sx(t), top + rh * len(rows) + 13, t, 8.2, "middle", MUTED)
    for i, (name, v, c) in enumerate(rows):
        y = top + i * rh
        b += text(L - 8, y + 13, name, 9, "end", style="italic" if c == GREY else "normal", fill=MUTED if c == GREY else INK)
        b += f'<path d="M {L} {y + 4} H {sx(v) - 3} a 3 3 0 0 1 3 3 v 6 a 3 3 0 0 1 -3 3 H {L} z" fill="{c}"/>'
        b += f'<rect x="{sx(v) + 2}" y="{y + 4}" width="30" height="12" fill="#fff"/>' + text(sx(v) + 5, y + 13, f"{v:.2f}", 9, weight="600")
    b += text((L + R) / 2, top + rh * len(rows) + 27, "Overall token F1 (%)", 8.6, "middle", MUTED)
    return svg(640, top + rh * len(rows) + 32, b)

# ---------------------------------------------------------------- figure 3: MaViLS per lecture
def fig3():
    rows = [("Phonetics", 0.99, 0.65), ("Solar resource", 0.81, 0.53), ("Climate &amp; Cities", 0.98, 0.86), ("Numerics", 0.89, 0.81),
            ("Computation theory", 0.87, 0.84), ("Deep learning", 0.99, 0.99), ("ML for health", 0.91, 0.95),
            ("Reinforcement", 0.67, 0.75), ("Short range", 0.58, 0.80)]
    L, R, top, rh = 130, 540, 34, 20
    sx = lambda v: L + (R - L) * (v - 0.4) / 0.6
    b = f'<circle cx="{L + 5}" cy="12" r="4.5" fill="{BLUE}"/>' + text(L + 14, 15.5, "Ours", 9)
    b += f'<rect x="{L + 55}" y="7.5" width="9" height="9" fill="#fff" stroke="{ORANGE}" stroke-width="2"/>' + text(L + 70, 15.5, "MaViLS, all features, published [3]", 9)
    n = len(rows) + 1
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += f'<line x1="{sx(t)}" y1="{top - 4}" x2="{sx(t)}" y2="{top + rh * n + 4}" stroke="{RULE}" stroke-width="0.5"/>' + text(sx(t), top + rh * n + 17, f"{t:.1f}", 8.2, "middle", MUTED)
    for i, (name, ours, theirs) in enumerate(rows + [("Mean of the nine", 0.853, 0.798)]):
        y = top + i * rh + rh / 2 + (6 if i == len(rows) else 0)
        mean = i == len(rows)
        if mean:
            b += f'<line x1="0" y1="{y - 13}" x2="640" y2="{y - 13}" stroke="{RULE}" stroke-width="0.6"/>'
        b += text(L - 10, y + 3.2, name, 9, "end", weight="600" if mean else "normal")
        b += f'<line x1="{sx(theirs)}" y1="{y}" x2="{sx(ours)}" y2="{y}" stroke="{GREY}" stroke-width="2"/>'
        b += f'<rect x="{sx(theirs) - 4.5}" y="{y - 4.5}" width="9" height="9" fill="#fff" stroke="{ORANGE}" stroke-width="2"/>'
        b += f'<circle cx="{sx(ours)}" cy="{y}" r="5" fill="{BLUE}" stroke="#fff" stroke-width="1.5"/>'
        d = ours - theirs
        lab = "tie" if abs(d) < 0.005 else (f"+{d:.3f}" if mean else f"{d:+.2f}".replace("-", "−"))
        b += text(R + 48, y + 3.2, lab, 9, "end", weight="600" if mean else "normal")
    b += text(R + 48, top - 8, "difference", 8.2, "end", MUTED)
    b += text((L + R) / 2, top + rh * n + 31, "F1 per lecture (authors' scorer)", 8.6, "middle", MUTED)
    return svg(640, top + rh * n + 36, b)

FIGS = {
 "fig1": (fig1(), "<b>Figure 1.</b> The LinkRAG pipeline. Dashed boxes are the link-centred components; they are the ones for which neither benchmark shows a benefit (§7). The baseline path is hybrid retrieval, the answerer and the verifier."),
 "fig2": (fig2(), "<b>Figure 2.</b> Overall token F1 on LectQA-Vid open-ended questions. Grey rows are references: a floor that needs no system, and the published score, which used an unnamed answering model. Every retrieval method lies between the echo floor and the full transcript."),
 "fig3": (fig3(), "<b>Figure 3.</b> MaViLS, the nine held-out lectures with video, ordered by difference. Our gains are on lectures where the published system scores low; our losses are on camera-heavy lectures. The mean difference is +0.055 with CI [−0.050, +0.161]."),
}
ANCHOR = {
 "fig1": "<p>Every stage runs in a baseline and a linked mode from the same code, so that each component is an ablation.</p>\n",
 "fig2": "<p>We recommend that results on LectQA-Vid be accompanied by a full-transcript row, a question-echo row, a\nclosed-book row and, for multiple choice, a longest-option row.</p>\n",
 "fig3": "against 0.53); the largest loss is on Short range (0.58 against 0.80).</p>\n",
}
p = "paper/paper.html"; t = open(p).read()
t = re.sub(r"<figure id=\"fig\d\">.*?</figure>\n", "", t, flags=re.S)
for k, (s, cap) in FIGS.items():
    assert t.count(ANCHOR[k]) == 1, k
    t = t.replace(ANCHOR[k], ANCHOR[k] + f'<figure id="{k}">{s}<figcaption>{cap}</figcaption></figure>\n')
if "figure {" not in t:
    t = t.replace(".todo { color: #8a1c1c; }", ".todo { color: #8a1c1c; }\nfigure { margin: 3mm 0 4mm; break-inside: avoid; }\nfigcaption { font-size: 8.8pt; line-height: 1.3; margin-top: 1.5mm; text-align: left; hyphens: none; }\nfigcaption b { font-family: Helvetica, Arial, sans-serif; }")
open(p, "w").write(t); print("figures inserted:", t.count("<figure"))
