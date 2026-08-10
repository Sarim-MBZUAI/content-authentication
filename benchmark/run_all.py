#!/usr/bin/env python3
"""Run every adapter in adapters/ over the manifest -> scores/<detector>.csv."""
import argparse, os, subprocess, sys, time

B = "/shared/home/sarim.hashmi/usenix/benchmark"
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--only", default=None, help="comma-list of adapter names")
ap.add_argument("--limit", default=None)
args = ap.parse_args()

adapters = sorted(f[:-3] for f in os.listdir(f"{B}/adapters") if f.endswith(".py"))
if args.only:
    adapters = [a for a in adapters if a in args.only.split(",")]
print(f"{len(adapters)} adapters: {adapters}", flush=True)

failed = []
for a in adapters:
    out = f"{B}/scores/{a}.csv"
    cmd = [f"{B}/venv/bin/python", f"{B}/adapters/{a}.py",
           "--manifest", f"{B}/manifest.csv", "--out", out, "--device", args.device]
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
