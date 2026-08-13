# Adapter contract (arms-race benchmark)

Each detector gets ONE standalone script `adapters/<detector>.py`.

## CLI (mandatory, exact)
```
python adapters/<detector>.py --manifest <csv> --out <csv> [--device cuda|cpu] [--limit N]
```
- `--manifest`: CSV with columns `generator,gen_year,label,path`.
- `--out`: output CSV path.
- `--device`: default `cuda`. Must also work on `cpu` (used for smoke tests only).
- `--limit N`: score only the first N manifest rows (smoke tests).

## Output CSV (mandatory, exact columns)
```
generator,gen_year,label,path,score
```
- `score` = float, **higher = more likely FAKE/generated** (P(fake) or any
  monotone score; flip sign if the repo's convention is opposite).
- One row per manifest row, same order. On a per-image error, write score
  as empty string and continue (do not crash the whole run).

## Rules
- Load the OFFICIAL checkpoint from
  `/shared/home/sarim.hashmi/usenix/detectors/<year>/<Name>/weights/`
  and official code from `.../code/` (add to sys.path; do not copy/modify the repo).
- Use each repo's own preprocessing (its test transform). No homogenization.
- Interpreter: `/shared/home/sarim.hashmi/usenix/benchmark/venv/bin/python`.
  Missing deps: `venv/bin/pip install <pkg>` — NEVER reinstall/up/downgrade torch.
- NO GPU on the login node (SLURM kills it). Smoke test with
  `--device cpu --limit 4` only.
- Batch where easy; float16 on cuda OK if the repo does it.
