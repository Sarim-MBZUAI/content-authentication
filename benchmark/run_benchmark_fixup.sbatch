#!/bin/bash
#SBATCH --job-name=armsrace-fix
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.armsrace_fix_slurm.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
MISSING=$(venv/bin/python -c "
import os
adapters = sorted(f[:-3] for f in os.listdir('adapters') if f.endswith('.py'))
missing = [a for a in adapters if not os.path.exists(f'scores/{a}.csv')]
print(','.join(missing))")
echo "rerunning: $MISSING"
[ -n "$MISSING" ] && venv/bin/python run_all.py --device cuda --manifest manifest_normalized.csv --scores-dir scores --only "$MISSING"
venv/bin/python analyze.py
venv/bin/python to_jsonl.py
