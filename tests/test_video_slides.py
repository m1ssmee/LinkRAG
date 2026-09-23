"""Frame-derived slide units: segmentation, transitions, revisits, figure boxes."""

from __future__ import annotations

from PIL import Image, ImageDraw

from linkrag.ingest.video_slides import dhash, figure_boxes, hamming, segment_slides, slide_at


def slide(kind: int) -> Image.Image:
    """Three visually distinct 'slides' (different bar layouts) on a white background."""
    img = Image.new("RGB", (320, 180), "white")
    d = ImageDraw.Draw(img)
    for i in range(4):
        x = 20 + ((i * 70 + kind * 37) % 260)
        d.rectangle([x, 20 + kind * 30, x + 30, 160 - kind * 20], fill=(0, 0, 0))
    return img


def test_dhash_separates_slides_and_tolerates_noise():
    a, b = slide(0), slide(1)
    noisy = a.copy()
    noisy.putpixel((5, 5), (200, 200, 200))
    assert hamming(dhash(a), dhash(noisy)) <= 2
    assert hamming(dhash(a), dhash(b)) > 10


def test_segments_transitions_and_revisits():
    # 1 fps: slide 0 for 5 s, a 1 s transition frame, slide 1 for 5 s, slide 0 again for 4 s
    frames = [(float(t), slide(0)) for t in range(0, 5)]
    frames += [(5.0, slide(2))]                                   # 1 s: a transition, not a slide
    frames += [(float(t), slide(1)) for t in range(6, 11)]
    frames += [(float(t), slide(0)) for t in range(11, 15)]
    slides = segment_slides(frames, max_dist=10, min_dur=2.0, step=1.0)
    assert len(slides) == 2                                       # transition merged, revisit deduplicated
    assert slides[0].intervals == [(0.0, 6.0), (11.0, 15.0)]      # the transition frame joined slide 0's run
    assert slides[1].intervals == [(6.0, 11.0)]
    assert slide_at(slides, 3.0) == 0 and slide_at(slides, 7.5) == 1 and slide_at(slides, 12.0) == 0
    assert slide_at(slides, 20.0) is None


def test_figure_boxes_find_the_figure_not_the_text_or_a_corner_logo():
    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([300, 100, 560, 300], outline=(0, 0, 0), width=3)          # a chart frame
    d.line([320, 280, 540, 120], fill=(0, 0, 0), width=3)                  # ... with a curve
    d.rectangle([610, 330, 630, 350], fill=(0, 0, 0))                      # a corner logo
    d.rectangle([40, 60, 220, 72], fill=(0, 0, 0))                         # a "text line"
    words = [(40, 60, 220, 72, "title")]                                   # OCR says it is text
    boxes = figure_boxes(img, words)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 300 and y0 <= 100 and x1 >= 560 and y1 >= 300 and x1 < 610
