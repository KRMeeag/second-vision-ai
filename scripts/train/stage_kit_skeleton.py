#!/usr/bin/env python3
"""
Build v2_kit_skeleton.tar.gz -- the deployment kit MINUS the two big items that
already live on the RunPod network volume.

WHY
---
Compiling over a mobile connection, the full kit is 1,199.9 MiB of upload
(500.1 wheel + 699.8 kit). But the network volume (DEC-137, survived the balance
lapse) already holds, verified:

  * all 6,644 calibration images in /workspace/bundle/images -- byte-identical to
    the kit's, because both are products of the same DEC-119 640-bounded resize;
  * best.pt at the exact expected sha256 ab3678b6995b2b02..., in two locations.

Those are 680.8 MiB and 21.5 MiB of the kit. Everything else -- 6,644 labels, the
nc=14 data.yaml, classes.yaml.frozen, provenance/pip_freeze/args, the eval
CSV/JSON, results.csv, RESTORE.md -- totals **0.90 MiB**. So the upload becomes
501.0 MiB instead of 1,199.9 MiB.

WHY NOT TAKE THE LABELS FROM THE VOLUME TOO
-------------------------------------------
The volume has /workspace/final_cap4500_v2/val/labels with the right count, but
proving it is the *same* label set as the kit's means trusting a path name. The
labels are 0.85 MiB. Shipping them from the verified kit costs nothing and removes
the question entirely -- which matters here because a label mix-up between v2a and
v2b is exactly the silent class of error DEC-127/DEC-135 exist to document.

So only two things are sourced from the volume, and both are checked by digest
rather than by trust. See scripts/train/assemble_kit_on_pod.py.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "models/current/deployment_kit_v2_cap4500"
OUT = REPO / "v2_kit_skeleton.tar.gz"

# What the assembled kit must prove on the pod. Both computed from the local kit.
EXPECT_PATHSIZE = "6490abf10110e411bded8a1169932d1cc479298102ac28c77baec5cce9993a5f"
EXPECT_CONTENT = "8a41a50ca4a3f634ad5caa8a09883b5371bfffe43e006d3a116d658afcbe8815"
EXPECT_FILES = 13297
WEIGHTS_SHA = "ab3678b6995b2b023d2cceb1370b7f7cf2fca86dede89eac1b96acbb6d327f73"

# Sourced from the volume, so excluded from the skeleton.
FROM_VOLUME_DIR = "calibration/images"
FROM_VOLUME_FILE = "best.pt"


def digests(kit: Path) -> tuple[str, str, int]:
    """Both digests over the kit, excluding CHECKSUM and RESTORE.md (as DEC-135)."""
    fs = [p for p in sorted(kit.rglob("*"))
          if p.is_file() and p.name not in ("CHECKSUM", "RESTORE.md")]
    ps, ct = hashlib.sha256(), hashlib.sha256()
    for p in fs:
        rel = str(p.relative_to(kit)).encode()
        ps.update(rel); ps.update(str(p.stat().st_size).encode())
        ct.update(rel); ct.update(p.read_bytes())
    return ps.hexdigest(), ct.hexdigest(), len(fs)


def main() -> None:
    if not KIT.is_dir():
        sys.exit(f"no kit at {KIT} -- build it with build_deployment_kit_v2.py")

    ps, ct, n = digests(KIT)
    problems = []
    if ps != EXPECT_PATHSIZE: problems.append(f"path+size digest is {ps}, expected {EXPECT_PATHSIZE}")
    if ct != EXPECT_CONTENT:  problems.append(f"content digest is {ct}, expected {EXPECT_CONTENT}")
    if n != EXPECT_FILES:     problems.append(f"{n} files, expected {EXPECT_FILES}")
    if problems:
        print("local kit does not match the recorded digests:", file=sys.stderr)
        for p in problems: print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    print(f"local kit verified: {n} files")
    print(f"  path+size {ps[:16]}...")
    print(f"  content   {ct[:16]}...")

    img_dir = KIT / FROM_VOLUME_DIR
    images = sorted(p.name for p in img_dir.iterdir() if p.is_file())

    members = [p for p in sorted(KIT.rglob("*")) if p.is_file()
               and FROM_VOLUME_DIR not in str(p.relative_to(KIT))
               and p.name != FROM_VOLUME_FILE]

    # The manifest is how the pod knows WHICH 6,644 images to pull off the volume
    # and what size each must be -- image extensions vary (.jpg/.jpeg/.png), so a
    # label-stem scan is not enough on its own.
    manifest = "\n".join(f"{(img_dir / nm).stat().st_size}  {nm}" for nm in images) + "\n"
    expect = (f"pathsize {EXPECT_PATHSIZE}\n"
              f"content {EXPECT_CONTENT}\n"
              f"files {EXPECT_FILES}\n"
              f"weights_sha256 {WEIGHTS_SHA}\n")

    if OUT.exists():
        OUT.unlink()
    with tarfile.open(OUT, "w:gz") as tf:
        for p in members:
            tf.add(p, arcname=f"v2_kit_skeleton/{p.relative_to(KIT)}")
        for name, text in (("MANIFEST_images.txt", manifest), ("EXPECT.txt", expect)):
            data = text.encode()
            info = tarfile.TarInfo(f"v2_kit_skeleton/{name}")
            info.size = len(data)
            import io
            tf.addfile(info, io.BytesIO(data))

    size = OUT.stat().st_size
    print(f"\nwrote {OUT.relative_to(REPO)}")
    print(f"  {len(members)} kit files + manifest ({len(images)} images listed) + EXPECT.txt")
    print(f"  {size:,} bytes ({size / 1048576:.2f} MiB)")
    print(f"\nupload total with the wheel: "
          f"{(size + (REPO / 'hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl').stat().st_size) / 1048576:.1f} MiB"
          f"  (vs 1199.9 MiB for the full kit)")


if __name__ == "__main__":
    main()
