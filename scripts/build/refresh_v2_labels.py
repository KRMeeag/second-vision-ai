"""
Refresh dataset/final_v2a and dataset/final_v2b label CONTENT in place
from the authoritative dataset/processed/<source>/labels/ trees.

The image set and the file set are NEVER changed — this rewrites the
contents of label files that already exist, so the v1-vs-v2 controlled
comparison and the frozen split both survive. A label file present in a
final tree but missing from processed/ is a hard error, not a skip.

final_v2a is the 15-class schema (Shelf at id 5).
final_v2b is the 14-class schema: Shelf dropped, every id > 5 shifted
down by one (DEC-126).

Usage:
    python3 scripts/build/refresh_v2_labels.py           # dry run
    python3 scripts/build/refresh_v2_labels.py --apply
"""
import sys, collections
from pathlib import Path

APPLY = "--apply" in sys.argv
PROC = Path("dataset/processed")
SOURCES = sorted((p.name for p in PROC.iterdir() if p.is_dir()), key=len, reverse=True)
SHELF_ID = 5

def split_name(stem):
    for s in SOURCES:
        if stem.startswith(s + "__"):
            return s, stem[len(s) + 2:]
    return None, None

def remap_v2b(lines):
    out = []
    for ln in lines:
        parts = ln.split()
        cid = int(parts[0])
        if cid == SHELF_ID:
            continue
        parts[0] = str(cid - 1 if cid > SHELF_ID else cid)
        out.append(" ".join(parts))
    return out

stats = collections.Counter()
missing = []
per_class_a = collections.Counter()
per_class_b = collections.Counter()
empty_after_shelf = []

for lf in sorted(Path("dataset/final_v2a").rglob("*/labels/*.txt")):
    stem = lf.stem
    src, base = split_name(stem)
    if src is None:
        missing.append((str(lf), "UNRECOGNISED SOURCE PREFIX")); continue
    srcfile = PROC / src / "labels" / f"{base}.txt"
    if not srcfile.exists():
        missing.append((str(lf), f"missing {srcfile}")); continue
    lines = [l.strip() for l in srcfile.read_text().split("\n") if l.strip()]
    for l in lines: per_class_a[int(l.split()[0])] += 1
    b_lines = remap_v2b(lines)
    for l in b_lines: per_class_b[int(l.split()[0])] += 1
    if lines and not b_lines:
        empty_after_shelf.append(stem)

    bf = Path(str(lf).replace("final_v2a", "final_v2b", 1))
    if not bf.exists():
        missing.append((str(bf), "v2b counterpart missing")); continue

    a_txt = ("\n".join(lines) + "\n") if lines else ""
    b_txt = ("\n".join(b_lines) + "\n") if b_lines else ""
    if lf.read_text() != a_txt:
        stats["v2a_changed"] += 1
        if APPLY: lf.write_text(a_txt)
    if bf.read_text() != b_txt:
        stats["v2b_changed"] += 1
        if APPLY: bf.write_text(b_txt)
    stats["files"] += 1

print(f"files processed:   {stats['files']}")
print(f"v2a files changed: {stats['v2a_changed']}")
print(f"v2b files changed: {stats['v2b_changed']}")
print(f"v2a total boxes:   {sum(per_class_a.values())}")
print(f"v2b total boxes:   {sum(per_class_b.values())}")
print(f"labelled-but-empty-after-Shelf-drop (background images): {len(empty_after_shelf)}")
if missing:
    print(f"\n!! {len(missing)} PROBLEMS:")
    for m in missing[:20]: print("   ", m)
    sys.exit(1)
print("\nv2a per-class:", dict(sorted(per_class_a.items())))
print("v2b per-class:", dict(sorted(per_class_b.items())))
print("\n(dry run — pass --apply to write)" if not APPLY else "\nAPPLIED")
