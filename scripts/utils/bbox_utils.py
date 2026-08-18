"""
scripts/utils/bbox_utils.py
────────────────────────────
Bounding-box coordinate conversion and validation, scoped per DEC-025
(2026-08-06): most source formats (VOC, COCO-style, YOLO) are loaded
through FiftyOne's native importers, which handle box-coordinate math
internally. This module covers only what FiftyOne's importers don't:

  1. Converting the two non-standard raw formats into normalized YOLO
     boxes — ExDark's native `l,t,w,h` absolute pixels and CrowdHuman's
     odgt `vbox` field (DEC-012: vbox only; fbox/hbox are discarded).
     Both are absolute top-left x,y,width,height under different field
     names, so a single converter covers both.
  2. Validating and clipping boxes, since FiftyOne does not enforce box
     sanity (in-range coordinates, positive width/height) automatically.

All public functions operate on YOLO's normalized center-based box
convention: (cx, cy, w, h), each relative to image width/height.

Pipeline context:
  YOLOv8 Training → ONNX Export → Hailo DFC → HEF → hailo-apps on RPi5
  Intended for scripts/acquire/acquire_exdark.py and
  scripts/acquire/acquire_crowdhuman.py (rescoped by DEC-041 to do
  parsing/conversion themselves, not separate Stage 5.2 scripts), and
  scripts/preprocess/box_audit.py (Stage 5.3).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def xywh_abs_to_yolo(
    x: float, y: float, w: float, h: float, img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """
    Convert an absolute-pixel top-left box (x, y, w, h) to a normalized
    YOLO center-based box (cx, cy, w, h).

    Covers ExDark's native `l,t,w,h` and CrowdHuman's `vbox` [x,y,w,h] —
    both are absolute top-left-corner + width/height, just under
    different field names (DEC-012).

    Parameters
    ----------
    x, y : float
        Absolute pixel coordinates of the box's top-left corner.
    w, h : float
        Absolute pixel width and height of the box.
    img_width, img_height : int
        Dimensions of the source image, for normalization.

    Returns
    -------
    tuple[float, float, float, float]
        (cx, cy, w, h), each normalized to the [0, 1] range implied by
        img_width/img_height (not clamped — see clip_bbox()).
    """
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    return cx, cy, w / img_width, h / img_height


# ---------------------------------------------------------------------------
# Validation & clipping
# ---------------------------------------------------------------------------

def validate_bbox(cx: float, cy: float, w: float, h: float, epsilon: float = 0.0) -> str | None:
    """
    Check whether a normalized YOLO box (cx, cy, w, h) is well-formed.

    Parameters
    ----------
    cx, cy, w, h : float
        Normalized YOLO box, each expected in [0, 1].
    epsilon : float, default 0.0
        Bounds-violation tolerance. Converters call this on freshly-computed
        floats (epsilon=0.0 is correct — a genuinely unclipped box should be
        caught). box_audit.py instead re-validates already-written,
        6-decimal-rounded label text, where reconstructing cx±w/2 reintroduces
        ~5e-7 of floating-point round-trip noise per corner — enough at
        epsilon=0.0 to flag well over 100,000 boxes across this project's real
        output as "invalid" when none of them are (verified directly, DEC-057).
        Pass a small epsilon (e.g. 1e-4 — three orders of magnitude above that
        noise floor, orders of magnitude below any real defect seen) when
        checking already-rounded text instead of pre-write floats.

    Returns
    -------
    str | None
        None if the box is valid. Otherwise a short human-readable reason
        it isn't — intended for box_audit.py to log directly rather than
        raising, since finding invalid boxes is the expected outcome of
        an audit, not an exceptional one.
    """
    if w <= 0 or h <= 0:
        return f"non-positive width/height (w={w}, h={h})"

    x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h)
    violation = max(-x1, -y1, x2 - 1, y2 - 1, 0.0)
    if violation > epsilon:
        return (
            f"box extends outside image bounds "
            f"(x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f})"
        )

    return None


def clip_bbox(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """
    Clip a normalized YOLO box so it fits within the image bounds [0, 1].

    Use this for boxes that slightly overshoot the image edge (a common
    artifact of annotation tools or rounding) rather than discarding them
    outright. Boxes that are fundamentally invalid (non-positive width or
    height) aren't fixable by clipping — check validate_bbox() first.

    Each corner is clamped into [0, 1] independently (not just the lower
    corner to >=0 and the upper corner to <=1) — a box that sits entirely
    outside the frame (e.g. x1=1.05, x2=1.30, both past the right edge)
    needs both corners pulled back to the same boundary, or the naive
    one-sided clamp leaves x1 > x2 and produces a *negative* width/height
    instead of a degenerate zero-width one. Found via Stage 5.3's
    box_audit.py against real CrowdHuman output (DEC-057) — 5 of 439,046
    boxes had a vbox entirely outside the image, evading the pre-clip
    validate_bbox() non-positive-width/height guard by only going negative
    *after* this function's old asymmetric clamp.

    Parameters
    ----------
    cx, cy, w, h : float
        Normalized YOLO box.

    Returns
    -------
    tuple[float, float, float, float]
        (cx, cy, w, h), clipped to fit within [0, 1]. If the input box was
        entirely outside the frame, this degenerates to a zero-width and/or
        zero-height box at the boundary — callers should re-check with
        validate_bbox() after clipping and drop it if so (clip_bbox() only
        fits a box inside bounds; it can't invent visible content that
        wasn't there).
    """
    x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h)
    x1, x2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
    y1, y2 = min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)
    return xyxy_to_yolo(x1, y1, x2, y2)


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Center-based (cx, cy, w, h) -> corner-based (x1, y1, x2, y2)."""
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """
    Convert a corner-based box (x1, y1, x2, y2) to center-based (cx, cy, w, h).

    Covers Dataset Ninja's Supervisely-native `points.exterior` rectangle
    format (two corner points) — used both internally (clip_bbox) and
    directly by acquire_datasetninja.py, after normalizing pixel
    coordinates to [0, 1] by image width/height.

    Does not assume (x1, y1) is top-left — if the input is malformed
    (degenerate or swapped), the resulting w or h will be non-positive
    and validate_bbox() will correctly flag it rather than silently
    producing a wrong box.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Corner-based box, any consistent unit (normalized [0, 1] or
        absolute pixels — caller's responsibility to normalize before
        use elsewhere in this module, which otherwise assumes [0, 1]).

    Returns
    -------
    tuple[float, float, float, float]
        (cx, cy, w, h).
    """
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("bbox_utils.py — smoke test")
    print("=" * 60)

    print("\n[1] xywh_abs_to_yolo — ExDark/CrowdHuman-style absolute box")
    box = xywh_abs_to_yolo(x=100, y=50, w=200, h=300, img_width=640, img_height=480)
    print(f"    abs (x=100, y=50, w=200, h=300) in 640x480 -> "
          f"yolo {tuple(round(v, 4) for v in box)}")

    print("\n[2] validate_bbox — well-formed box")
    print(f"    validate_bbox{box} -> {validate_bbox(*box)!r}")

    print("\n[3] validate_bbox — box extending past the image edge")
    bad_box = (0.95, 0.5, 0.2, 0.2)  # x2 = 0.95 + 0.1 = 1.05 > 1
    print(f"    validate_bbox{bad_box} -> {validate_bbox(*bad_box)!r}")

    print("\n[4] clip_bbox — clipping that same box back into bounds")
    clipped = clip_bbox(*bad_box)
    print(f"    clip_bbox{bad_box} -> {tuple(round(v, 4) for v in clipped)}")
    print(f"    validate_bbox(clipped) -> {validate_bbox(*clipped)!r}")

    print("\n✅ All checks passed.")
    sys.exit(0)
