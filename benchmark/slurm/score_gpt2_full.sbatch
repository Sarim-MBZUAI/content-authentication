#!/bin/bash
#SBATCH --job-name=gpt2-19
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.score_gpt2_full.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
venv/bin/python run_all.py --device cuda --manifest manifest_gpt2_full.csv --scores-dir scores_gpt2_full
echo ALL DONE
