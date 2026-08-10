#!/usr/bin/env bash
# Queue FLUX.1 Boreal on physical GPU 0 after the FLUX.2 Boreal tmux job.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_SESSION="flux2boreal_gpu0"
CACHE_ROOT="$ROOT/.hf-boreal-flux1-cache/models--black-forest-labs--FLUX.1-dev"
LOG="$ROOT/logs/flux1boreal_gpu0.log"
STATUS="$ROOT/logs/flux1boreal_gpu0.status"
PID_FILE="$ROOT/logs/flux1boreal_gpu0.pid"
mkdir -p "$ROOT/logs" "$ROOT/out/flux1boreal/fake"

set_status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >"$STATUS"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >>"$LOG"
}

snapshot_ready() {
  local commit snapshot tree
  [[ -s "$CACHE_ROOT/refs/main" ]] || return 1
  commit="$(tr -d '\r\n' <"$CACHE_ROOT/refs/main")"
  snapshot="$CACHE_ROOT/snapshots/$commit"
  tree="$CACHE_ROOT/trees/$commit.json"
  [[ -d "$snapshot" && -s "$tree" ]] || return 1
  python3 - "$tree" "$snapshot" <<'PY'
import json
import sys
from pathlib import Path

tree = json.loads(Path(sys.argv[1]).read_text())
snapshot = Path(sys.argv[2])
for name, metadata in tree["files"].items():
    path = snapshot / name
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        raise SystemExit(1)
PY
}

set_status "waiting_for_flux2boreal_gpu0_to_appear"
while ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do sleep 5; done
set_status "waiting_for_flux2boreal_gpu0_to_exit"
while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do sleep 15; done
set_status "waiting_for_complete_flux1_dev_snapshot"
while ! snapshot_ready; do sleep 30; done

commit="$(tr -d '\r\n' <"$CACHE_ROOT/refs/main")"
model="$CACHE_ROOT/snapshots/$commit"
set_status "starting"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

set +e
"$ROOT/.venv-flux2-clean/bin/python" "$ROOT/generation/gen_flux1_boreal.py" \
  --prompts "$ROOT/generation/prompts.csv" \
  --out "$ROOT/out/flux1boreal/fake" \
  --model "$model" \
  --lora "$ROOT/.hf-boreal-flux1-adapter/boreal-v2.safetensors" \
  --trigger photo --lora-scale 0.7 --seed 42 --size 1024 \
  --steps 28 --guidance 3.5 >>"$LOG" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" >"$PID_FILE"
set_status "running pid=$worker_pid"
wait "$worker_pid"
rc=$?
set -e

if (( rc == 0 )); then
  count="$(find "$ROOT/out/flux1boreal/fake" -maxdepth 1 -type f -name '*.png' | wc -l)"
  set_status "complete images=$count"
else
  set_status "failed exit_code=$rc"
fi
exit "$rc"
