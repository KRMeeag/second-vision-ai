"""
Generate missing-box proposals over the cap9000 EXTRA images (cap9000 - cap4500).
Companion to generate_recovery_proposals.py; writes a SEPARATE file so the
cap4500 results (65,172 proposals) are never truncated.

Two independent detectors vote:
  * yolov8x  (COCO-pretrained) - can propose 8 of our classes, and is INDEPENDENT
    of our training data, so it sees objects our own model is blind to.
  * cap4500_yolov8s_v2-2      - our 14-class model, covers the 6 classes COCO lacks.

A proposal is emitted only when NO existing label of the same class already covers
that region (IoU >= 0.5). Each carries a tier so review can proceed in batches:

    A  x_conf >= 0.70 and v2 confirms     highest confidence
    B  x_conf >= 0.70, v2 silent
    C  x_conf in [0.50,0.70) and v2 confirms
    D  x_conf in [0.50,0.70), v2 silent
    E  x_conf in [0.25,0.50)              lowest; review hardest

Also flagged (never auto-accepted):
    nested=true   the proposal sits inside, or swallows, an existing box of ANY
                  class at IoMin >= 0.9 - the DEC-133 hierarchy-duplicate signature.

Writes JSONL; changes NO labels. Promotion is a separate, reviewed step.
"""
import os, json, sys
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from pathlib import Path
from ultralytics import YOLO

N15=["Person","Vehicle","Motorcycle","Pole","Animals","Shelf","Doors","Chairs","Tables",
     "Tricycle","Potholes","Trash Bins","Bicycle","Stairs","Bench"]
N14=[n for i,n in enumerate(N15) if i!=5]
C2O={0:"Person",1:"Bicycle",2:"Vehicle",3:"Motorcycle",5:"Vehicle",7:"Vehicle",13:"Bench",
     14:"Animals",15:"Animals",16:"Animals",17:"Animals",18:"Animals",19:"Animals",
     20:"Animals",21:"Animals",22:"Animals",23:"Animals",56:"Chairs",57:"Chairs",60:"Tables"}
ROOT=Path(__file__).resolve().parents[2]
PROC=ROOT/"dataset"/"processed"; BUNDLE=ROOT/"dataset"/"bundle"/"images"
OUT=ROOT/"dataset"/"proposals"/"recovery_cap9000_extra.jsonl"
SRC=sorted((p.name for p in PROC.iterdir() if p.is_dir()), key=len, reverse=True)

def src_of(e):
    for s in SRC:
        if e.startswith(s+"__"): return s, e[len(s)+2:]
    return None, None
def gt(e):
    s,n=src_of(e)
    p=PROC/s/"labels"/f"{n}.txt" if s else None
    o=[]
    if p and p.exists():
        for l in p.read_text().splitlines():
            q=l.split()
            if len(q)==5: o.append((N15[int(q[0])], tuple(map(float,q[1:]))))
    return o
def _inter(a,b):
    return (max(0,min(a[0]+a[2]/2,b[0]+b[2]/2)-max(a[0]-a[2]/2,b[0]-b[2]/2))
          * max(0,min(a[1]+a[3]/2,b[1]+b[3]/2)-max(a[1]-a[3]/2,b[1]-b[3]/2)))
def iou(a,b):
    i=_inter(a,b); u=a[2]*a[3]+b[2]*b[3]-i
    return i/u if u>0 else 0.0
def iomin(a,b):
    i=_inter(a,b); m=min(a[2]*a[3], b[2]*b[3])
    return i/m if m>0 else 0.0
def tier(xc, agree):
    if xc>=0.70: return "A" if agree else "B"
    if xc>=0.50: return "C" if agree else "D"
    return "E"

def main():
    MAN=ROOT/"dataset"/"bundle"/"manifests"
    m9=json.loads((MAN/"cap9000.json").read_text())
    m4=json.loads((MAN/"cap4500.json").read_text())
    p9=set(m9["train"])|set(m9["val"])|set(m9["test"])
    p4=set(m4["train"])|set(m4["val"])|set(m4["test"])
    pool=sorted(p9-p4)   # the 11,344 train-only images cap4500 omits
    idx={}
    for p in BUNDLE.iterdir():
        if p.is_file():
            idx[p.name[:p.name.rfind(p.suffix)] if p.suffix else p.name]=p
    pool=[e for e in pool if e in idx]
    X=YOLO(str(ROOT/"notebooks"/"yolov8x.pt"))
    V=YOLO(str(ROOT/"runs"/"detect"/"cap4500_yolov8s_v2-2"/"weights"/"best.pt"))
    n=0
    with OUT.open("w") as fh:
        for i in range(0,len(pool),32):
            ch=pool[i:i+32]; ps=[idx[e] for e in ch]
            rx=X.predict(ps,conf=0.25,device="mps",verbose=False)
            rv=V.predict(ps,conf=0.10,device="mps",verbose=False)
            for e,a,b in zip(ch,rx,rv):
                g=gt(e)
                vb=([(N14[int(c)],tuple(x),cf) for x,c,cf in
                     zip(b.boxes.xywhn.tolist(),b.boxes.cls.tolist(),b.boxes.conf.tolist())]
                    if b.boxes is not None else [])
                if a.boxes is None: continue
                for box,c,cf in zip(a.boxes.xywhn.tolist(),a.boxes.cls.tolist(),a.boxes.conf.tolist()):
                    cl=C2O.get(int(c))
                    if not cl: continue
                    box=tuple(box)
                    if max([iou(box,gg[1]) for gg in g if gg[0]==cl]+[0])>=0.5: continue
                    vm=max([(iou(box,v[1]),v[2]) for v in vb if v[0]==cl]+[(0,0)])
                    agree=vm[0]>=0.5
                    nested=any(iomin(box,gg[1])>=0.9 for gg in g)
                    fh.write(json.dumps({"image":e,"cls":cl,"box":[round(v,6) for v in box],
                                         "x_conf":round(cf,4),"v2_conf":round(vm[1],4),
                                         "agree":agree,"nested":nested,
                                         "tier":tier(cf,agree)})+"\n")
                    n+=1
            if (i//32)%40==0:
                print(f"  {i+len(ch):>6,}/{len(pool):,} images  {n:,} proposals", flush=True)
    print(f"DONE {n:,} proposals over {len(pool):,} images -> {OUT}")

if __name__=="__main__":
    main()
