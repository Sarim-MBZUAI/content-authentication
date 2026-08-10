#!/bin/bash
#SBATCH --job-name=flux2-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=gen_flux2_%j.log
source venv/bin/activate
python generation/gen_flux2.py --prompts generation/prompts.csv --out out/FLUX.2/fake
