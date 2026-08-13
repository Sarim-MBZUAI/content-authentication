#!/bin/bash
#SBATCH --job-name=armsrace-norm
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.armsrace_norm_slurm.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
venv/bin/python run_all.py --device cuda --manifest manifest_normalized.csv --scores-dir scores
venv/bin/python analyze.py
venv/bin/python to_jsonl.py
