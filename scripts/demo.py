"""make demo -- smoke run proving the data model and config wire together.

Note: offline and fake on purpose (no API key, no corpus, no model
download). Replace the hand-built units with a real ingest->index->link->
retrieve pass as those modules land; the printing below stays the same.
"""

from __future__ import annotations

from dataclasses import asdict

from linkrag.core import EvidenceUnit, Link, Location, load_config
from linkrag.generate.answer import format_evidence

cfg = load_config()

units = [
    EvidenceUnit(
        id="u1",
        modality="audio",
        content="...and as you can see here, the attention weights concentrate on the subject token.",
        source_file="data/raw/lecture03.wav",
        location=Location(start_s=612.0, end_s=628.5),
    ),
    EvidenceUnit(
        id="u2",
        modality="figure",
        content="Heatmap of attention weights, layer 6 head 3; brightest cell on the subject token.",
        source_file="data/raw/slides03.pdf",
        location=Location(page=14, bbox=(72.0, 310.0, 520.0, 640.0)),
    ),
    EvidenceUnit(
        id="u3",
        modality="text",
        content="Self-attention assigns each token a weight over all other tokens in the sequence.",
        source_file="data/raw/notes.docx",
        location=Location(page=2),
    ),
]

links = [
    Link("u1", "u2", "deictic_visual", 0.81),   # "here" -> the heatmap on screen
    Link("u2", "u3", "figure_paragraph", 0.64),  # figure -> the prose defining it
]

print(f"mode={cfg['mode']}  embedder={cfg['models']['embedding']}  top_k={cfg['retrieve']['top_k']}\n")
print(format_evidence(units))
print("\nLinks built by the Evidence Linking Layer:")
for link in links:
    print(f"  {link.src_id} -[{link.link_type} {link.score:.2f}]-> {link.dst_id}")

# Baseline would return u3 alone (best plain text match). LinkRAG follows
# deictic_visual and figure_paragraph to hand the LLM all three.
seed = "u1"
reached = {seed} | {l.dst_id for l in links if l.src_id == seed}
reached |= {l.dst_id for l in links if l.src_id in reached}
print(f"\nbaseline evidence set: {{'u3'}}  ->  linkrag from seed {seed}: {sorted(reached)}")
print(f"\nasdict round-trip ok: {asdict(units[0])['location']['start_s'] == 612.0}")
