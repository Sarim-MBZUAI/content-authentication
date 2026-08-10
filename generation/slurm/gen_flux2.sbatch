#!/bin/bash
#SBATCH --job-name=flux2-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --output=gen_flux2_%j.log
# NOTE: 300G RAM — the CPU-offload path keeps the ~110 GB bf16 pipeline in system memory.
# HF_TOKEN is NOT inherited into batch jobs by default — provide it here.
[ -z "$HF_TOKEN" ] && [ -f "$HOME/.env" ] && export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' "$HOME/.env" | tr -d "'\" ")
[ -n "$HF_TOKEN" ] || { echo "FATAL: HF_TOKEN not set in job environment"; exit 1; }
source venv/bin/activate
python generation/gen_flux2.py --prompts generation/prompts.csv --out out/FLUX.2/fake
