"""The Evidence Linking Layer -- novel component 1.

Builds typed Links at indexing time: audio_slide, figure_text, deictic,
same_slide.

baseline: no-op, returns an empty link set (what P1/P2/P3 effectively have).
          For alignment specifically, the baseline ablation is `align_naive`:
          per-segment argmax with no sequence structure, roughly P2's behaviour.
linkrag:  full link construction with per-type scoring. Implemented so far:
          - audio_slide      monotonic DP over the whole lecture (Phase 2)
          - figure_text      numbered/descriptive references, bbox layout
                             proximity, page prior and semantics (Phase 3)
          - deictic          resolved by composing with the audio_slide alignment,
                             so "as you can see here" reaches the figure that was on
                             screen rather than any figure in the corpus (Phase 3)
"""
