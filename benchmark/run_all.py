#!/usr/bin/env python3
"""Run every adapter in benchmark/adapters/ over a manifest -> <scores-dir>/<detector>.csv.
Run from the repository root."""
import argparse, os, subprocess, sys, time

B = "benchmark"
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{B}/manifest_normalized.csv",
                help="CSV with columns generator,gen_year,label,path (output of normalize_images.py)")
ap.add_argument("--scores-dir", default=f"{B}/scores")
ap.add_argument("--device", default="cuda")
ap.add_argument("--only", default=None, help="comma-list of adapter names")
ap.add_argument("--limit", default=None)
args = ap.parse_args()

adapters = sorted(f[:-3] for f in os.listdir(f"{B}/adapters") if f.endswith(".py"))
if args.only:
    adapters = [a for a in adapters if a in args.only.split(",")]
print(f"{len(adapters)} adapters: {adapters}", flush=True)
os.makedirs(args.scores_dir, exist_ok=True)

failed = []
for a in adapters:
    out = os.path.join(args.scores_dir, f"{a}.csv")
    cmd = [sys.executable, f"{B}/adapters/{a}.py",
           "--manifest", args.manifest, "--out", out, "--device", args.device]
    if args.limit:
        cmd += ["--limit", args.limit]
    t0 = time.time()
    print(f"=== {a} ===", flush=True)
    r = subprocess.run(cmd)
    status = "ok" if r.returncode == 0 and os.path.exists(out) else "FAILED"
    if status == "FAILED":
        failed.append(a)
    print(f"=== {a}: {status} ({time.time()-t0:.0f}s) ===", flush=True)

print(f"done. failed: {failed or 'none'}", flush=True)
sys.exit(1 if failed else 0)
