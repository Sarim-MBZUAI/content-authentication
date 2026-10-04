#!/usr/bin/env python3
"""Kill the resolution confound: every manifest image -> shorter side 512,
center-crop 512x512, PNG, into normalized/<generator>/<real|fake>/<name>.
Writes manifest_normalized.csv with identical rows but normalized paths.
Idempotent: existing outputs are skipped."""
import csv, os
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

B = Path("benchmark")
SIZE = 512

def normalize(row):
    src = Path(row["path"])
    sub = "real" if row["label"] == "0" else "fake"
    dst = B / "normalized" / row["generator"] / sub / (src.stem + ".png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        img = Image.open(src).convert("RGB")
        w, h = img.size
        s = SIZE / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = img.size
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        img.crop((l, t, l + SIZE, t + SIZE)).save(dst)
    return {**row, "path": str(dst)}

rows = list(csv.DictReader(open(B / "manifest.csv")))
with ThreadPoolExecutor(16) as ex:
    out = list(ex.map(normalize, rows))
with open(B / "manifest_normalized.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["generator", "gen_year", "label", "path"])
    w.writeheader(); w.writerows(out)
# sanity
sizes = {Image.open(r["path"]).size for r in out[::97]}
print(f"normalized {len(out)} images; sampled sizes: {sizes}")
