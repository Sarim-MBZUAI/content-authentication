#!/bin/bash
#SBATCH --job-name=armsrace
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.armsrace_slurm.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
cd /shared/home/sarim.hashmi/usenix/benchmark
venv/bin/python extend_manifest.py
venv/bin/python run_all.py --device cuda
venv/bin/python analyze.py
