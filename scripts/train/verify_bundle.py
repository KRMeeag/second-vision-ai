"""
Verify a bundle after transfer. Run this ON THE POD before training.

WHAT THIS CATCHES
-----------------
A 5 GB transfer over a ~0.7 MB/s link takes ~2 h and will be resumed at least
once. The failure modes that matter are all silent:

  * a truncated final file -- tar extracts what it has and exits non-zero, but a
    resumed-then-forgotten transfer can leave a short image that only fails when
    the dataloader reaches it, potentially an hour into epoch 1
  * macOS AppleDouble junk (`._name.jpg`) extracted alongside the real files.
    `data/base.py:167` globs `**/*.*`, so those WOULD be picked up as images.
    The bundle was packed with COPYFILE_DISABLE=1 to prevent this; verifying it
    is how we know the prevention worked.
  * a manifest naming an image that did not survive the trip

THE DIGEST
----------
`prepare_runpod_bundle.py` wrote a sha256 over `name + size` for every image, in
`sorted(images.iterdir())` order. This recomputes it through the SAME expression
rather than a re-derivation, so the two cannot drift apart. It is a name+size
digest, not a content hash: it is ~200x faster than hashing 5 GB and it catches
the failure modes above (truncation changes size; junk files change the file
list). It would not catch a bit-flip inside an intact-length JPEG -- that is what
the separate `bundle.tar.sha256` covers, checked before extraction.

Usage
-----
    python3 scripts/train/verify_bundle.py --bundle /workspace/bundle
    python3 scripts/train/verify_bundle.py --bundle /workspace/bundle --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_digest(img_dir: Path) -> tuple[str, int]:
    """Byte-identical to prepare_runpod_bundle.py's CHECKSUM block.

    Deliberately does NOT filter dotfiles -- iterdir() picks up anything an
    extraction dropped in here, and an unexpected file SHOULD break the digest.
    """
    h = hashlib.sha256()
    n = 0
    for p in sorted(img_dir.iterdir()):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
        n += 1
    return h.hexdigest(), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Digest only; skip the per-manifest file-existence sweep.")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    img_dir, lbl_dir = bundle / "images", bundle / "labels"
    for p in (img_dir, lbl_dir, bundle / "CHECKSUM", bundle / "bundle_meta.json"):
        if not p.exists():
            print(f"FAIL: missing from bundle: {p}")
            return 1

    meta = json.loads((bundle / "bundle_meta.json").read_text(encoding="utf-8"))
    expected = (bundle / "CHECKSUM").read_text(encoding="utf-8").split()[0]

    print(f"bundle : {bundle}")
    print(f"nc     : {meta['nc']}  ({len(meta['names'])} names)")
    print(f"built  : {meta['created']}  git {meta.get('git_sha', '?')[:12]}")

    print("\ncomputing name+size digest ...")
    got, n_img = compute_digest(img_dir)
    ok = got == expected
    print(f"  images found : {n_img}   (expected {meta['superset_images']})")
    print(f"  expected     : {expected}")
    print(f"  computed     : {got}")
    print(f"  digest       : {'MATCH' if ok else 'MISMATCH'}")

    failures = []
    if not ok:
        failures.append("image digest does not match CHECKSUM")
    if n_img != meta["superset_images"]:
        failures.append(f"image count {n_img} != {meta['superset_images']}")

    # AppleDouble / hidden junk, reported separately because the digest alone
    # only says "something differs", not what.
    junk = [p.name for p in img_dir.iterdir() if p.name.startswith(".")]
    if junk:
        failures.append(f"{len(junk)} hidden file(s) in images/, e.g. {junk[:3]} "
                        f"-- ultralytics WILL try to load these; delete them")

    n_lbl = sum(1 for p in lbl_dir.iterdir() if p.suffix == ".txt")
    print(f"  labels found : {n_lbl}")
    if n_lbl != meta["superset_images"]:
        failures.append(f"label count {n_lbl} != {meta['superset_images']}")

    if not args.quick:
        print("\nchecking manifests ...")
        stems = {p.stem for p in img_dir.iterdir() if p.is_file()}
        lbls = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}
        for mf in sorted((bundle / "manifests").glob("cap*.json")):
            m = json.loads(mf.read_text(encoding="utf-8"))
            tr, va, te = set(m["train"]), set(m["val"]), set(m["test"])
            miss_i = (tr | va | te) - stems
            miss_l = (tr | va | te) - lbls
            leak = tr & (va | te)
            status = "OK" if not (miss_i or miss_l or leak) else "FAIL"
            print(f"  {mf.stem:<9} train {len(tr):>6} val {len(va):>5} test {len(te):>5}"
                  f"   missing img {len(miss_i)}  missing lbl {len(miss_l)}"
                  f"  train/eval overlap {len(leak)}   [{status}]")
            if miss_i:
                failures.append(f"{mf.stem}: {len(miss_i)} manifest image(s) absent, "
                                f"e.g. {sorted(miss_i)[:2]}")
            if miss_l:
                failures.append(f"{mf.stem}: {len(miss_l)} manifest label(s) absent")
            if leak:
                failures.append(f"{mf.stem}: {len(leak)} image(s) in BOTH train and eval")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nDo not train on this bundle. Re-transfer, then re-run this check.")
        return 1

    print("\nPASS - bundle is intact and every manifest resolves.")
    print("Next: python3 scripts/train/materialize_condition.py "
          f"--bundle {bundle} --cap 4500")
    return 0


if __name__ == "__main__":
    sys.exit(main())
