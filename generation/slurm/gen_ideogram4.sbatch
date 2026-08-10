#!/bin/bash
#SBATCH --job-name=ideo4-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=gen_ideogram4_%j.log
# HF_TOKEN is NOT inherited into batch jobs by default — provide it here.
# Either submit with `sbatch --export=ALL,HF_TOKEN=$HF_TOKEN ...` or keep it in ~/.env:
[ -z "$HF_TOKEN" ] && [ -f "$HOME/.env" ] && export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' "$HOME/.env" | tr -d "'\" ")
[ -n "$HF_TOKEN" ] || { echo "FATAL: HF_TOKEN not set in job environment"; exit 1; }
source venv/bin/activate
python generation/gen_ideogram4.py --prompts generation/prompts.csv --out out/Ideogram4/fake
