"""Pre-flight for the v3 run. Fails LOUDLY before any GPU time is spent.

Every check here corresponds to something that has already gone wrong on this
project: a wrong ultralytics version, symlinks that did not resolve, a dataset whose
label paths silently mismatched, and a volume whose exhausted quota produced 0-byte
files instead of errors.
"""
import json, os, shutil, subprocess, sys
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from pathlib import Path

ARMS = sys.argv[1:] or ["v3p", "v3l"]
PIN = "8.4.118"
fail = []

def ok(m):   print(f"  OK    {m}", flush=True)
def bad(m):  print(f"  FAIL  {m}", flush=True); fail.append(m)

print("=== pre-flight ===", flush=True)

# 1. versions -- the pin matters: v2 was trained on 8.4.118 and the comparison
#    is only meaningful if the arms use the same.
try:
    import ultralytics, torch
    (ok if ultralytics.__version__ == PIN else bad)(
        f"ultralytics {ultralytics.__version__} (pin {PIN})")
    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        ok(f"GPU {torch.cuda.get_device_name(0)} "
           f"{torch.cuda.get_device_properties(0).total_memory//2**20} MiB")
    else:
        bad("no CUDA device visible")
except Exception as e:
    bad(f"import failed: {e}")

# 2. each arm resolves through ultralytics' OWN loader, not our parsing
for arm in ARMS:
    d = Path(f"/workspace/{arm}/data.yaml")
    if not d.is_file():
        bad(f"{arm}: missing {d}"); continue
    try:
        from ultralytics.data.utils import check_det_dataset
        spec = check_det_dataset(str(d), autodownload=False)
        n = {}
        for split in ("train", "val", "test"):
            p = Path(spec[split])
            lines = [l for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []
            n[split] = len(lines)
            miss = sum(1 for l in lines[:200] if not Path(l).is_file())
            if miss: bad(f"{arm}/{split}: {miss}/200 sampled image paths do not resolve")
        ok(f"{arm}: nc={spec['nc']} train={n['train']:,} val={n['val']:,} test={n['test']:,}")
        lab = Path(f"/workspace/{arm}/labels")
        ok(f"{arm}: {len(list(lab.iterdir())):,} label files")
    except Exception as e:
        bad(f"{arm}: dataset check failed: {e}")

# 3. val/test must be IDENTICAL across arms or the comparison is meaningless
import hashlib
sig = {}
for arm in ARMS:
    for split in ("val", "test"):
        p = Path(f"/workspace/{arm}/{split}.txt")
        if p.is_file():
            sig.setdefault(split, {})[arm] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
for split, d in sig.items():
    (ok if len(set(d.values())) == 1 else bad)(
        f"{split}.txt identical across arms: {d}")

# 4. writable headroom on the volume -- an exhausted quota here writes 0-byte files
#    rather than failing, which is how this project lost artefacts before.
try:
    probe = Path("/workspace/.preflight_probe")
    need = 3 * 1024  # MiB, ~3 GiB covers checkpoints for three arms
    with probe.open("wb") as f:
        f.write(b"\0" * (need * 1024 * 1024))
    got = probe.stat().st_size // (1024 * 1024)
    probe.unlink()
    (ok if got >= need else bad)(f"volume headroom: wrote {got} MiB of {need} MiB")
except Exception as e:
    bad(f"volume write probe failed: {e}")
    try: Path("/workspace/.preflight_probe").unlink()
    except Exception: pass

# 5. comet, if configured -- reported, never fatal
if os.environ.get("COMET_API_KEY"):
    try:
        import comet_ml  # noqa: F401
        ok(f"comet_ml present, project={os.environ.get('COMET_PROJECT_NAME','(default)')}")
    except ImportError:
        print("  WARN  COMET_API_KEY set but comet_ml not installed -- no Comet logging", flush=True)
else:
    print("  WARN  COMET_API_KEY not set -- training will not log to Comet", flush=True)

print(f"\n{'PRE-FLIGHT FAILED' if fail else 'PRE-FLIGHT PASSED'}", flush=True)
for f in fail: print(f"  - {f}", flush=True)
sys.exit(1 if fail else 0)
