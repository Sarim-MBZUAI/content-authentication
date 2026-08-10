# liars-dividend-generation

Private research repo: image generation + inversion code for the
plausible-deniability / content-authentication project (USENIX '26).

- `SKILL.md` — **start here**: runbook to generate the 100-image FLUX.2-dev
  (2025) and Ideogram 4 (2026) benchmark sets on any GPU machine.
- `generation/` — portable generation scripts, the fixed 100-prompt set
  (`prompts.csv`), and SLURM templates.
- `inversion/` — RF-Inversion-based invert+resynthesize pipelines for
  SD2.1 / SD3 / SD3.5 plus evaluation code.
- `rf-inversion/` — upstream RF-Inversion reference implementation.

Model weights are NOT in this repo (HF-gated, see SKILL.md). No tokens or
secrets are stored here — set `HF_TOKEN` in the environment.
