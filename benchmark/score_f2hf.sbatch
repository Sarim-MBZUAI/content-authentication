#!/bin/bash
#SBATCH --job-name=score-f2hf
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.score_f2.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
for d in SICA D3 OmniAID; do echo "=== $d ==="; venv/bin/python adapters/$d.py --manifest manifest_f2hf.csv --out scores_f2hf_$d.csv --device cuda; done
echo DONE
