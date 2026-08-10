#!/usr/bin/env bash
# Downloads Ideogram-4-fp8 + FLUX.2-dev weights DIRECTLY from Hugging Face.
# Needs: HF_TOKEN env var (gates for both repos already accepted on the account).
# Resumable — rerun on any failure. Logs to download_weights.log.

DEST="/home/lukas/users/shashmi/liars-dividend-generation"
LOG="$(cd "$(dirname "$0")" && pwd)/download_weights.log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[ERROR] line $LINENO: command failed: $BASH_COMMAND (see $LOG)"' ERR
set -eE

echo "===== $(date) — download_weights.sh start ====="
[ -n "$HF_TOKEN" ] || { echo "[ERROR] set HF_TOKEN first: export HF_TOKEN=hf_..."; exit 1; }
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

python3 - <<PY
from huggingface_hub import snapshot_download
base = "$DEST"
print("-> ideogram-ai/ideogram-4-fp8 (~28 GB)")
snapshot_download("ideogram-ai/ideogram-4-fp8", local_dir=f"{base}/ideogram-4-fp8", max_workers=8)
print("-> black-forest-labs/FLUX.2-dev (~113 GB, skipping single-file duplicate)")
snapshot_download("black-forest-labs/FLUX.2-dev", local_dir=f"{base}/FLUX.2-dev",
                  ignore_patterns=["flux2-dev.safetensors"], max_workers=8)
print("downloads complete")
PY

echo "----- verify -----"
du -sh "$DEST/ideogram-4-fp8" "$DEST/FLUX.2-dev"
echo "===== $(date) — done ====="
