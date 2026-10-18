#!/usr/bin/env python3
"""
Assemble the v2 deployment kit ON THE POD from the skeleton upload plus the two
large items already on the RunPod network volume. Verifies the result against
digests computed on the Mac, and refuses to produce a kit that does not match.

Run on the pod:

    python3 assemble_kit_on_pod.py --skeleton /root/v2_kit_skeleton.tar.gz \
                                   --out /root/v2_kit

WHAT COMES FROM WHERE, AND WHY
------------------------------
From the skeleton (uploaded, 0.79 MiB):   6,644 labels, data.yaml, the frozen
    nc=14 schema, provenance/pip_freeze/args, the eval CSV/JSON, results.csv,
    RESTORE.md, plus MANIFEST_images.txt and EXPECT.txt.
From the volume (not uploaded, 702 MiB):  the 6,644 calibration images and best.pt.

Labels are deliberately NOT taken from the volume even though
/workspace/final_cap4500_v2/val/labels has the right count: proving it is the same
label set would mean trusting a path name, and a v2a/v2b label mix-up is exactly
the silent error class DEC-127/DEC-135 document. They are 0.85 MiB, so shipping
them removes the question for free.

THE VERIFICATION IS THE POINT
-----------------------------
Three independent checks, all fatal:

  1. best.pt sha256 must equal the expected value exactly.
  2. The assembled tree's **path+size** digest must equal the kit's CHECKSUM --
     the same digest shape DEC-135 defined, so the kit stays interchangeable.
  3. The assembled tree's **content** digest must match. Path+size alone can pass
     on a same-length-different-bytes file; this closes that.

If all three hold, the kit on the pod is bit-for-bit what was verified on the Mac,
and sourcing the images from the volume introduced nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

# Candidate locations on the volume, tried in order.
IMAGE_DIRS = ("/workspace/bundle/images",)
WEIGHT_PATHS = (
    "/workspace/second-vision-ai/runs/detect/cap4500_yolov8s_v2-2/weights/best.pt",
    "/workspace/model_backups/cap4500_yolov8s_v2-2_*/weights/best.pt",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digests(kit: Path) -> tuple[str, str, int]:
    fs = [p for p in sorted(kit.rglob("*"))
          if p.is_file() and p.name not in ("CHECKSUM", "RESTORE.md")]
    ps, ct = hashlib.sha256(), hashlib.sha256()
    for p in fs:
        rel = str(p.relative_to(kit)).encode()
        ps.update(rel); ps.update(str(p.stat().st_size).encode())
        ct.update(rel); ct.update(p.read_bytes())
    return ps.hexdigest(), ct.hexdigest(), len(fs)


def find_images_dir(cands: list[str], probe: str) -> Path | None:
    for d in cands:
        p = Path(d)
        if p.is_dir() and (p / probe).is_file():
            return p
    return None


def find_weights(cands: list[str], want_sha: str) -> tuple[Path | None, list[str]]:
    tried = []
    for pat in cands:
        matches = sorted(Path("/").glob(pat.lstrip("/"))) if "*" in pat else [Path(pat)]
        for m in matches:
            if not m.is_file():
                continue
            got = sha256_file(m)
            tried.append(f"{m}  sha256 {got[:16]}...")
            if got == want_sha:
                return m, tried
    return None, tried


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skeleton", default="/root/v2_kit_skeleton.tar.gz")
    ap.add_argument("--out", default="/root/v2_kit")
    ap.add_argument("--images-dir", action="append", default=[],
                    help="override/add a calibration image source (repeatable)")
    ap.add_argument("--weights", action="append", default=[],
                    help="override/add a best.pt location (repeatable)")
    args = ap.parse_args()

    skel_tar, out = Path(args.skeleton), Path(args.out)
    if not skel_tar.is_file():
        sys.exit(f"no skeleton at {skel_tar}")

    # 1. unpack the skeleton into place
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    staging = out.parent / ".skel_unpack"
    if staging.exists():
        shutil.rmtree(staging)
    with tarfile.open(skel_tar) as tf:
        tf.extractall(staging)
    root = staging / "v2_kit_skeleton"
    if not root.is_dir():
        sys.exit("skeleton tar does not contain v2_kit_skeleton/")

    expect = dict(line.split(None, 1) for line in
                  (root / "EXPECT.txt").read_text().strip().splitlines())
    expect = {k: v.strip() for k, v in expect.items()}
    want_ps, want_ct = expect["pathsize"], expect["content"]
    want_n, want_wsha = int(expect["files"]), expect["weights_sha256"]

    manifest = []
    for line in (root / "MANIFEST_images.txt").read_text().splitlines():
        if line.strip():
            size, name = line.split(None, 1)
            manifest.append((int(size), name.strip()))
    print(f"skeleton: {len(manifest)} images expected, target {want_n} files")

    for item in sorted(root.rglob("*")):
        if item.name in ("MANIFEST_images.txt", "EXPECT.txt"):
            continue
        rel = item.relative_to(root)
        dst = out / rel
        if item.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
    print("  skeleton unpacked")

    # 2. best.pt from the volume, by checksum not by path
    wpath, tried = find_weights(args.weights + list(WEIGHT_PATHS), want_wsha)
    if wpath is None:
        print("\nFATAL: no best.pt on the volume matched the expected sha256", file=sys.stderr)
        print(f"  wanted {want_wsha}", file=sys.stderr)
        for t in tried:
            print(f"  tried  {t}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(wpath, out / "best.pt")
    print(f"  best.pt <- {wpath}  (sha256 verified)")

    # 3. the 6,644 calibration images from the volume
    img_out = out / "calibration/images"
    img_out.mkdir(parents=True, exist_ok=True)
    src = find_images_dir(args.images_dir + list(IMAGE_DIRS), manifest[0][1])
    if src is None:
        print(f"\nFATAL: no calibration image source contained {manifest[0][1]}", file=sys.stderr)
        print(f"  looked in: {', '.join(args.images_dir + list(IMAGE_DIRS))}", file=sys.stderr)
        sys.exit(1)
    missing, wrong = [], []
    for size, name in manifest:
        s = src / name
        if not s.is_file():
            missing.append(name); continue
        if s.stat().st_size != size:
            wrong.append(f"{name}: {s.stat().st_size} != {size}"); continue
        shutil.copy2(s, img_out / name)
    if missing or wrong:
        print(f"\nFATAL: calibration images from {src} are not usable", file=sys.stderr)
        if missing: print(f"  {len(missing)} missing, e.g. {missing[:3]}", file=sys.stderr)
        if wrong:   print(f"  {len(wrong)} wrong size, e.g. {wrong[:3]}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(manifest)} calibration images <- {src}")

    shutil.rmtree(staging)

    # 4. the three checks
    ps, ct, n = digests(out)
    ok = True
    print("\nverification")
    for label, got, want in (("files    ", str(n), str(want_n)),
                             ("path+size", ps, want_ps),
                             ("content  ", ct, want_ct)):
        good = got == want
        ok &= good
        mark = "OK  " if good else "FAIL"
        shown = got if label.strip() == "files" else f"{got[:16]}..."
        print(f"  {mark} {label}  {shown}")
        if not good and label.strip() != "files":
            print(f"       expected {want[:16]}...")
    if not ok:
        print("\nKit does NOT match the Mac. Do not compile from it.", file=sys.stderr)
        sys.exit(1)

    (out / "CHECKSUM").write_text(
        f"{ps}  path+size digest over {n} files (excludes CHECKSUM, RESTORE.md)\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nkit assembled and verified: {out}  ({total / 1048576:.1f} MiB, {n + 2} files)")
    print("bit-for-bit identical to the kit verified on the Mac.")


if __name__ == "__main__":
    main()
