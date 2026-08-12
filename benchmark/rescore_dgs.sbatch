#!/bin/bash
#SBATCH --job-name=dgs-fix
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.dgs_fix.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
venv/bin/python adapters/DGS-Net.py --manifest manifest_new_gens_abs.csv --out scores_new_gens/DGS-Net.csv --device cuda
