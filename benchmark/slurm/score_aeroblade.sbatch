#!/bin/bash
#SBATCH --job-name=aeroblade
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/shared/home/sarim.hashmi/usenix/benchmark/.score_aeroblade.log
export HF_TOKEN=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' /shared/home/sarim.hashmi/.env | tr -d "'\" ")
export HF_HOME=/shared/home/sarim.hashmi/.cache/huggingface
cd /shared/home/sarim.hashmi/usenix/benchmark
P=venv/bin/python
$P adapters/AEROBLADE.py --manifest manifest_normalized.csv   --out scores/AEROBLADE.csv          --device cuda && echo "=== 5-gens done ==="
$P adapters/AEROBLADE.py --manifest manifest_new_gens.csv     --out scores_new_gens/AEROBLADE.csv --device cuda && echo "=== new-gens done ==="
$P adapters/AEROBLADE.py --manifest manifest_flux2_full.csv   --out scores_flux2/AEROBLADE.csv    --device cuda && echo "=== flux2 done ==="
$P adapters/AEROBLADE.py --manifest manifest_gpt2_full.csv    --out scores_gpt2_full/AEROBLADE.csv --device cuda && echo "=== gpt2 done ==="
echo ALL DONE
