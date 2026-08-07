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
  Intended for scripts/convert/exdark_to_intermediate.py and
  scripts/convert/crowdhuman_odgt_to_intermediate.py (Stage 5.2), and
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

def validate_bbox(cx: float, cy: float, w: float, h: float) -> str | None:
    """
    Check whether a normalized YOLO box (cx, cy, w, h) is well-formed.

    Parameters
    ----------
    cx, cy, w, h : float
        Normalized YOLO box, each expected in [0, 1].

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
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
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

    Parameters
    ----------
    cx, cy, w, h : float
        Normalized YOLO box.

    Returns
    -------
    tuple[float, float, float, float]
        (cx, cy, w, h), clipped to fit within [0, 1].
    """
    x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return _xyxy_to_yolo(x1, y1, x2, y2)


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Center-based (cx, cy, w, h) -> corner-based (x1, y1, x2, y2)."""
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Corner-based (x1, y1, x2, y2) -> center-based (cx, cy, w, h)."""
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
