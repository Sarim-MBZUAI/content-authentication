#!/usr/bin/env bash
# Run ON the lukas machine. Pulls Ideogram-4 + FLUX.2-dev weights from the cluster.
# Everything (stdout+stderr) is logged to pull_weights.log next to this script.

DEST="/home/lukas/users/shashmi/liars-dividend-generation"
SRC_HOST="sarim.hashmi@10.67.33.23"
SRC_DIR="/shared/home/sarim.hashmi/usenix/generators"
LOG="$(cd "$(dirname "$0")" && pwd)/pull_weights.log"

# log everything to file AND terminal
exec > >(tee -a "$LOG") 2>&1

trap 'echo "[ERROR] line $LINENO: command failed: $BASH_COMMAND (see $LOG)"' ERR
set -eE

echo "===== $(date) — pull_weights.sh start ====="
echo "dest: $DEST"
echo "src : $SRC_HOST:$SRC_DIR"

command -v rsync >/dev/null || { echo "[ERROR] rsync not installed on this machine"; exit 1; }
mkdir -p "$DEST"

for M in ideogram-4-fp8 FLUX.2-dev; do
    echo "----- pulling $M -----"
    rsync -avP --partial --stats --human-readable \
        "$SRC_HOST:$SRC_DIR/$M" "$DEST/" \
        || { echo "[ERROR] rsync failed for $M — rerun this script, it resumes."; exit 1; }
done

echo "----- verify -----"
du -sh "$DEST/ideogram-4-fp8" "$DEST/FLUX.2-dev"
echo "===== $(date) — done ====="
