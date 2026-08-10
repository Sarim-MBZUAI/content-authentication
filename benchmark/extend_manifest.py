#!/usr/bin/env python3
"""Extend manifest.csv with FLUX.2 (2025) and Ideogram4 (2026) rows when their
image sets are complete (100 real + 100 fake each). Idempotent; safe to rerun."""
import csv, os

B = "/shared/home/sarim.hashmi/usenix/inverted_stuff"
MAN = "/shared/home/sarim.hashmi/usenix/benchmark/manifest.csv"
NEW = {"FLUX.2": (f"{B}/FLUX.2", 2025), "Ideogram4": (f"{B}/Ideogram4", 2026)}

rows = list(csv.DictReader(open(MAN)))
have = {r["generator"] for r in rows}
added = 0
for gen, (root, yr) in NEW.items():
    if gen in have:
        continue
    reals = sorted(os.listdir(f"{root}/real")) if os.path.isdir(f"{root}/real") else []
    fakes = sorted(os.listdir(f"{root}/fake")) if os.path.isdir(f"{root}/fake") else []
    if len(reals) < 100 or len(fakes) < 100:
        print(f"{gen}: not ready ({len(reals)} real / {len(fakes)} fake)")
        continue
    for f in reals[:100]:
        rows.append({"generator": gen, "gen_year": yr, "label": 0, "path": f"{root}/real/{f}"})
    for f in fakes[:100]:
        rows.append({"generator": gen, "gen_year": yr, "label": 1, "path": f"{root}/fake/{f}"})
    added += 200
    print(f"{gen}: added 200 rows")

if added:
    with open(MAN, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["generator", "gen_year", "label", "path"])
        w.writeheader(); w.writerows(rows)
print(f"manifest now {len(rows)} rows")
