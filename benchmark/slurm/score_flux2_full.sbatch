#!/bin/bash
#SBATCH --job-name=f2-full19
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.score_flux2_full.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
venv/bin/python run_all.py --device cuda --manifest manifest_flux2_full.csv --scores-dir scores_flux2
echo ALL DONE
