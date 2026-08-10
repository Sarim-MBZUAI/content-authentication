#!/bin/bash
#SBATCH --job-name=ideo4-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=gen_ideogram4_%j.log
source venv/bin/activate
python generation/gen_ideogram4.py --prompts generation/prompts.csv --out out/Ideogram4/fake
