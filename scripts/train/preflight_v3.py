"""Pre-flight for the v3 three-arm run. Fails LOUDLY before any GPU time is spent.

Every check here corresponds to something that has already gone wrong on this
project or on this volume. None of it is hypothetical:

  * a data.yaml emitted by string surgery that was not valid YAML at all
  * a stale schema that rendered every Pothole as a Tricycle (DEC-107)
  * a data.yaml that resolved from its own directory and nowhere else (DEC-066)
  * symlinks that archived as links and calibrated INT8 on zero images (DEC-135)
  * an exhausted MooseFS quota that wrote 0-BYTE FILES instead of failing (DEC-142)
  * a Comet key that was present, reported as armed, and logged nothing

THE CHECK THAT MATTERS MOST
---------------------------
#7: the val and test LABEL CONTENTS, not just the split listings, hashed per arm and
required identical. The entire experiment is "same yardstick, different training
labels". If the yardstick moves, the comparison measures nothing, and a listing-only
check would not notice -- the filenames stay the same while the boxes inside change.
That is also why #9 refuses to proceed past a labels.cache: ultralytics keys that
cache on the PATH LIST, so rewritten label contents under identical filenames look
valid to it and it would serve the previous arm's boxes.

EVERY STAGED ARM IS VERIFIED, NOT JUST THE ONE BEING TRAINED
-----------------------------------------------------------
Check #7 -- val and test label contents identical ACROSS arms -- is meaningless with
one arm in hand, and it is precisely what makes an arm trained next month comparable
to one trained today. So this script verifies every arm present on the volume and
treats `--train` as reporting only. Scanning an arm nobody trains today costs about a
minute; discovering after the fact that two runs were scored on different labels costs
the experiment.

Usage
-----
    python3 scripts/train/preflight_v3.py                   # verify every staged arm
    python3 scripts/train/preflight_v3.py --train "v3p"     # ...and note v3p is today's
    python3 scripts/train/preflight_v3.py v3p v3l           # verify only these
    python3 scripts/train/preflight_v3.py --quick           # sample labels instead of all
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/.ultralytics")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROOT = Path("/workspace")
PIN = "8.4.118"
EXPECT_SPLITS = {"train": 30123, "val": 6644, "test": 6403}
EXPECT_LABELS = 43170
NEED_MIB = 3 * 1024          # ~3 GiB covers checkpoints for three arms
V2_WEIGHTS = ROOT / "model_backups" / "cap4500_yolov8s_v2-2_20260913T164555Z" / "weights" / "best.pt"

ALL_ARMS = ["v3p", "v3l", "v3s"]

QUICK = "--quick" in sys.argv
# --train "v3p v3l" names what is being TRAINED today. It does not narrow what is
# VERIFIED: see the module docstring.
TRAIN: list[str] = []
_skip: set[int] = set()
if "--train" in sys.argv:
    i = sys.argv.index("--train")
    if i + 1 < len(sys.argv):
        TRAIN = sys.argv[i + 1].split()
        # Skip by INDEX, not by value. --train "v3p v3l" arrives as ONE argv element
        # whose text ("v3p v3l") matches no element of TRAIN, so a value-based filter
        # let it through as a single bogus arm name.
        _skip.add(i + 1)
_explicit = [a for n, a in enumerate(sys.argv) if n and n not in _skip
             and not a.startswith("--")]
# Bare positional arms narrow verification deliberately (a one-off re-check);
# otherwise verify every arm actually staged on the volume.
ARMS = _explicit or [a for a in ALL_ARMS if (ROOT / a / "data.yaml").is_file()] or ALL_ARMS

fail: list[str] = []
warned: list[str] = []


print(f"verifying arms : {' '.join(ARMS)}", flush=True)
print(f"training today  : {' '.join(TRAIN) if TRAIN else '(not stated)'}", flush=True)
if TRAIN and set(TRAIN) - set(ARMS):
    print(f"  NOTE: {sorted(set(TRAIN) - set(ARMS))} is to be trained but is not staged "
          f"on the volume", flush=True)


def ok(m):   print(f"  OK    {m}", flush=True)
def bad(m):  print(f"  FAIL  {m}", flush=True); fail.append(m)
def warn(m): print(f"  WARN  {m}", flush=True); warned.append(m)
def sec(m):  print(f"\n--- {m} ---", flush=True)


# ── 1. the volume ───────────────────────────────────────────────────────────────
sec("1. /workspace is the network volume")
try:
    import subprocess
    df = subprocess.run(["df", "-h", str(ROOT)], capture_output=True, text=True).stdout
    dev = df.strip().splitlines()[-1].split()[0] if df else "?"
    if "overlay" in dev:
        bad(f"/workspace is an OVERLAY filesystem ({dev}) -- the volume is not attached. "
            f"Weights would die with the pod.")
    else:
        ok(f"/workspace backed by {dev}")
except Exception as e:
    warn(f"could not inspect /workspace ({e})")

# ── 2. versions ─────────────────────────────────────────────────────────────────
sec("2. pinned stack")
try:
    import torch

    import ultralytics
    (ok if ultralytics.__version__ == PIN else bad)(
        f"ultralytics {ultralytics.__version__} (pin {PIN})")
    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        ok(f"GPU {p.name} {p.total_memory // 2**20} MiB")
    else:
        bad("no CUDA device visible -- training would run on CPU at ~1/50 speed "
            "and report no error")
except Exception as e:
    bad(f"import failed: {e}")

# ── 3. ultralytics output paths point at the VOLUME ─────────────────────────────
sec("3. ultralytics writes to the volume")
# The default runs_dir is the RELATIVE string 'runs'. Launched from anywhere but the
# repo root, that puts six hours of weights on the container disk.
try:
    from ultralytics.utils import SETTINGS
    for key in ("runs_dir", "weights_dir", "datasets_dir"):
        v = str(SETTINGS.get(key, ""))
        (ok if v.startswith(str(ROOT)) else bad)(f"{key} = {v or '(unset)'}")
    ycd = os.environ.get("YOLO_CONFIG_DIR", "")
    (ok if ycd.startswith(str(ROOT)) else warn)(
        f"YOLO_CONFIG_DIR = {ycd or '(unset)'}"
        + ("" if ycd.startswith(str(ROOT)) else " -- settings will not survive a pod restart"))
except Exception as e:
    bad(f"could not read ultralytics SETTINGS: {e}")

# ── 4. the repo's own schema ────────────────────────────────────────────────────
sec("4. repo schema")
try:
    from scripts.utils.config_loader import CANONICAL_NAMES, EXPECTED_NC
    ok(f"config/classes.yaml: nc={EXPECTED_NC} {CANONICAL_NAMES}")
except Exception as e:
    bad(f"could not load the repo schema: {e}")
    print("\nPRE-FLIGHT FAILED (no schema to check against)", flush=True)
    sys.exit(1)

# ── 5. each arm, resolved through ultralytics' OWN loader, from an unrelated CWD ─
sec("5. each arm resolves (from /tmp, per DEC-066)")
# Running the resolution from /tmp is the point: a data.yaml that only works from
# its own directory passes locally and fails on the pod.
import yaml  # noqa: E402

cwd0 = Path.cwd()
os.chdir("/tmp")
try:
    from ultralytics.data.utils import check_det_dataset
except Exception as e:
    check_det_dataset = None
    bad(f"cannot import check_det_dataset: {e}")

arm_ok: list[str] = []
for arm in ARMS:
    d = ROOT / arm / "data.yaml"
    if not d.is_file():
        bad(f"{arm}: missing {d}")
        continue

    # 5a. the file is valid YAML and agrees with the repo. train.py checks this too;
    #     checking it here means a broken arm costs nothing instead of a pod session.
    try:
        spec_raw = yaml.safe_load(d.read_text(encoding="utf-8"))
    except Exception as e:
        bad(f"{arm}: data.yaml is not valid YAML: {e}")
        continue
    if not isinstance(spec_raw, dict) or "names" not in spec_raw:
        bad(f"{arm}: data.yaml has no 'names' key -- "
            f"keys present: {sorted(spec_raw) if isinstance(spec_raw, dict) else type(spec_raw)}")
        continue
    got_names = [spec_raw["names"][i] for i in sorted(spec_raw["names"])]
    if got_names != CANONICAL_NAMES or spec_raw.get("nc") != EXPECTED_NC:
        bad(f"{arm}: schema mismatch\n        data.yaml nc={spec_raw.get('nc')} {got_names}\n"
            f"        repo      nc={EXPECTED_NC} {CANONICAL_NAMES}")
        continue
    if "path" in spec_raw:
        bad(f"{arm}: data.yaml carries a 'path' key -- DEC-066 forbids it")
        continue
    ok(f"{arm}: data.yaml valid, nc={EXPECTED_NC}, names match, no 'path' key")

    # 5b. ultralytics' own resolution
    if check_det_dataset is None:
        continue
    try:
        spec = check_det_dataset(str(d), autodownload=False)
    except Exception as e:
        bad(f"{arm}: check_det_dataset failed: {e}")
        continue

    counts = {}
    bad_split = False
    for split in ("train", "val", "test"):
        p = Path(spec[split])
        if not p.is_file():
            bad(f"{arm}/{split}: split listing {p} is not a file")
            bad_split = True
            continue
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        counts[split] = len(lines)
        if len(lines) != EXPECT_SPLITS[split]:
            bad(f"{arm}/{split}: {len(lines):,} entries, expected {EXPECT_SPLITS[split]:,}")
            bad_split = True
        # the images must actually be there, and the label-path rule must hold
        probe = lines[:: max(1, len(lines) // 200)][:200]
        miss_img = [l for l in probe if not Path(l).is_file()]
        if miss_img:
            bad(f"{arm}/{split}: {len(miss_img)}/{len(probe)} sampled images do not exist "
                f"(first: {miss_img[0]})")
            bad_split = True
        # ultralytics derives the label path by replacing the LAST '/images/'
        miss_lab = [l for l in probe
                    if not Path("/labels/".join(l.rsplit("/images/", 1)))
                            .with_suffix(".txt").is_file()]
        if miss_lab:
            bad(f"{arm}/{split}: {len(miss_lab)}/{len(probe)} sampled images have no label "
                f"at the path ultralytics derives (first: {miss_lab[0]})")
            bad_split = True
    if not bad_split:
        ok(f"{arm}: splits train={counts['train']:,} val={counts['val']:,} "
           f"test={counts['test']:,} (exact), sampled images+labels resolve")

    # 5c. the images symlink
    link = ROOT / arm / "images"
    tgt = link.resolve() if link.exists() else None
    if tgt == ROOT / "bundle" / "images" and tgt.is_dir():
        ok(f"{arm}: images -> {tgt}")
    else:
        bad(f"{arm}: images symlink resolves to {tgt}, expected {ROOT/'bundle'/'images'}")

    # 5d. label file count
    lab = ROOT / arm / "labels"
    try:
        n = sum(1 for _ in lab.iterdir())
        (ok if n == EXPECT_LABELS else bad)(
            f"{arm}: {n:,} label files (expect {EXPECT_LABELS:,})")
    except Exception as e:
        bad(f"{arm}: cannot list {lab}: {e}")

    arm_ok.append(arm)
os.chdir(cwd0)

# ── 6. label CONTENTS are well formed ───────────────────────────────────────────
# ── 7. val/test contents identical across arms; train contents all distinct ─────
sec("6-7. label contents: well formed, and the yardstick does not move")
t0 = time.time()
digests: dict[str, dict[str, str]] = {}
for arm in arm_ok:
    lab = ROOT / arm / "labels"
    per_split: dict[str, str] = {}
    issues = Counter()
    first_issue = {}
    cls_hist = Counter()
    for split in ("train", "val", "test"):
        p = ROOT / arm / f"{split}.txt"
        stems = [Path(l).name.rsplit(".", 1)[0]
                 for l in p.read_text().split() if l.strip()]
        # --quick samples the VALIDATION of train labels, never the digest: the
        # digest is what proves the yardstick has not moved, so it is always
        # computed over every file.
        scan = (set(stems[:: max(1, len(stems) // 3000)])
                if (QUICK and split == "train") else None)
        h = hashlib.sha256()
        for stem in sorted(stems):          # sorted: the digest must not depend on listing order
            f = lab / f"{stem}.txt"
            try:
                b = f.read_bytes()
            except FileNotFoundError:
                issues["label missing"] += 1
                first_issue.setdefault("label missing", str(f))
                continue
            h.update(stem.encode())
            h.update(b)
            if scan is not None and stem not in scan:
                continue
            if b and not b.endswith(b"\n"):
                issues["no trailing newline"] += 1
                first_issue.setdefault("no trailing newline", str(f))
            for ln, line in enumerate(b.decode("utf-8", "replace").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    issues["not 5 fields"] += 1
                    first_issue.setdefault("not 5 fields", f"{f}:{ln} {line!r}")
                    continue
                try:
                    c = int(parts[0])
                    xywh = [float(v) for v in parts[1:]]
                except ValueError:
                    issues["unparseable numbers"] += 1
                    first_issue.setdefault("unparseable numbers", f"{f}:{ln} {line!r}")
                    continue
                cls_hist[c] += 1
                if not 0 <= c < EXPECTED_NC:
                    issues[f"class id out of range (0..{EXPECTED_NC-1})"] += 1
                    first_issue.setdefault("class id out of range", f"{f}:{ln} id={c}")
                if any(not 0.0 <= v <= 1.0 for v in xywh):
                    issues["coord outside [0,1]"] += 1
                    first_issue.setdefault("coord outside [0,1]", f"{f}:{ln} {xywh}")
                if xywh[2] <= 0 or xywh[3] <= 0:
                    issues["zero or negative w/h"] += 1
                    first_issue.setdefault("zero or negative w/h", f"{f}:{ln} {xywh}")
        per_split[split] = h.hexdigest()[:16]
    digests[arm] = per_split
    if issues:
        for k, n in issues.items():
            bad(f"{arm}: {n:,} x {k} (first: {first_issue.get(k.split(' (')[0], '?')})")
    else:
        if not cls_hist:
            bad(f"{arm}: scanned every label and found ZERO boxes -- "
                f"empty or 0-byte label files (the MooseFS quota failure mode, DEC-142)")
        else:
            ok(f"{arm}: labels well formed"
               + (" (train sampled, --quick)" if QUICK else "")
               + f"; class ids {min(cls_hist)}..{max(cls_hist)}, "
                 f"{sum(cls_hist.values()):,} boxes scanned")

for split in ("val", "test"):
    vals = {a: digests[a][split] for a in arm_ok if a in digests}
    if len(vals) < 2:
        warn(f"{split}: only {len(vals)} arm available, so 'identical across arms' was "
             f"NOT tested. Any arm trained later is comparable to this one only if that "
             f"arm is staged from the same build -- stage all arms together.")
    elif len(set(vals.values())) == 1:
        ok(f"{split} label CONTENTS identical across {len(vals)} arms "
           f"({next(iter(vals.values()))})")
    else:
        bad(f"{split} label contents DIFFER across arms -- the evaluation yardstick "
            f"moves and no arm is comparable to another: {vals}")

train_d = {a: digests[a]["train"] for a in arm_ok if a in digests}
if len(train_d) < 2:
    warn(f"train: only {len(train_d)} arm available, so 'distinct per arm' was not tested")
elif len(set(train_d.values())) == len(train_d):
    ok(f"train label contents distinct per arm: {train_d}")
else:
    bad(f"two arms share identical TRAINING labels -- one of them is not the arm it "
        f"claims to be: {train_d}")
print(f"        (label scan took {time.time()-t0:.0f}s)", flush=True)

# ── 8. box counts match what the runbook and DECISIONS claim ─────────────────────
sec("8. training box counts match the recorded figures")
EXPECT_BOXES = {"v3s": 113965, "v3p": 118279, "v3l": 140801}
for arm in arm_ok:
    lab = ROOT / arm / "labels"
    stems = [Path(l).name.rsplit(".", 1)[0]
             for l in (ROOT / arm / "train.txt").read_text().split() if l.strip()]
    n = 0
    for stem in stems:
        try:
            n += sum(1 for l in (lab / f"{stem}.txt").read_text().splitlines() if l.strip())
        except FileNotFoundError:
            pass
    exp = EXPECT_BOXES.get(arm)
    if exp is None:
        warn(f"{arm}: {n:,} train boxes (no recorded figure to check against)")
    elif n == exp:
        ok(f"{arm}: {n:,} train boxes (matches the recorded figure)")
    else:
        bad(f"{arm}: {n:,} train boxes, but {exp:,} is what DECISIONS.md and the runbook "
            f"claim -- one of them is wrong and the write-up would be too")

# ── 9. no stale labels.cache ────────────────────────────────────────────────────
sec("9. no stale labels.cache")
# Ultralytics hashes the image+label PATH LIST, not the contents. Re-staging an arm
# rewrites label CONTENTS under identical filenames, so a surviving cache looks valid
# and would serve the PREVIOUS arm's boxes for the whole run.
# Scoped to the arms being TRAINED. A cache under an arm nobody is training today
# is harmless, and hard-failing on it would block the second arm entirely: after
# v3p trains it legitimately leaves /workspace/v3p/labels.cache, and since this
# script verifies EVERY staged arm, `run_v3.sh v3l` would then refuse to start.
hot = TRAIN or ARMS
hot_found, cold_found = [], []
for arm in ARMS:
    for c in (ROOT / arm).glob("*.cache"):
        if c.is_file():
            (hot_found if arm in hot else cold_found).append(str(c))
if hot_found:
    bad("labels.cache present for an arm being trained -- ultralytics keys that "
        "cache on the image+label PATH LIST and on total size, never on contents "
        "(utils.py get_hash), so labels rewritten under identical filenames look "
        "valid to it. setup_pod_v3.sh renames these aside:\n        "
        + "\n        ".join(hot_found))
else:
    ok(f"no .cache under the arm(s) being trained ({' '.join(hot)})")
if cold_found:
    print(f"  INFO  {len(cold_found)} .cache file(s) under arms not being trained "
          f"today -- harmless, left alone", flush=True)

# ── 10. writable headroom ───────────────────────────────────────────────────────
sec("10. volume headroom")
# An exhausted quota on this MooseFS volume writes 0-byte files rather than failing,
# which is how artefacts were lost before (DEC-142). `df` reports the whole cluster,
# never the quota, so the only honest test is to write.
probe = ROOT / ".preflight_probe"
try:
    with probe.open("wb") as f:
        f.write(b"\0" * (NEED_MIB * 1024 * 1024))
    got = probe.stat().st_size // (1024 * 1024)
    (ok if got >= NEED_MIB else bad)(f"wrote {got} MiB of {NEED_MIB} MiB probe")
except Exception as e:
    bad(f"volume write probe failed: {e}")
finally:
    try:
        probe.unlink()
    except Exception:
        pass

# ── 11. the v2 reference points ─────────────────────────────────────────────────
sec("11. v2 reference points (non-fatal)")
ref = REPO / "scripts" / "train" / "v2_args_reference.yaml"
(ok if ref.is_file() else warn)(
    f"hyperparameter reference {ref}" + ("" if ref.is_file() else " MISSING -- parity cannot be checked"))
if V2_WEIGHTS.is_file():
    ok(f"v2 weights for the re-baseline: {V2_WEIGHTS}")
else:
    warn(f"v2 weights not at {V2_WEIGHTS} -- the baseline phase will be skipped; the "
         f"arms stay comparable to each other, just not to where the project stands")

# ── 12. comet ───────────────────────────────────────────────────────────────────
sec("12. Comet (non-fatal -- this is how you check in remotely)")
key = os.environ.get("COMET_API_KEY")
if not key:
    warn("COMET_API_KEY not set -- no metric streaming; you would have to read --status")
else:
    try:
        import socket

        import comet_ml
        prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            comet_ml.API(api_key=key).get_workspaces()
            ok(f"Comet key AUTHENTICATES, project={os.environ.get('COMET_PROJECT_NAME','(default)')}")
        except Exception as e:
            # A key that exists is not a key that works. This exact failure logs
            # nothing for six hours and reports no error.
            warn(f"COMET_API_KEY is set but did NOT authenticate ({type(e).__name__}) -- "
                 f"training will log to TensorBoard only")
        finally:
            socket.setdefaulttimeout(prev)
    except ImportError:
        warn("COMET_API_KEY set but comet_ml is not installed")

# ── verdict ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PRE-FLIGHT FAILED" if fail else "PRE-FLIGHT PASSED")
for f_ in fail:
    print(f"  FAIL  {f_.splitlines()[0]}")
for w in warned:
    print(f"  WARN  {w.splitlines()[0]}")
print("=" * 70, flush=True)
sys.exit(1 if fail else 0)
