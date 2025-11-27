#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o /exports/eddie/scratch/s1891340/look-ahead-bias/logs/server-logs/steered-humaneval-qwen2_5.out
#$ -e /exports/eddie/scratch/s1891340/look-ahead-bias/logs/server-logs/steered-humaneval-qwen2_5.err

cd /exports/eddie/scratch/s1891340/look-ahead-bias
source ~/.bashrc
source /exports/csce/eddie/inf/groups/FinComputing/waylon/venv/look-ahead/bin/activate
export PYTHONPATH=PYTHONPATH:./

uv run --active --no-sync python benchmark/humaneval/eval.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 2048 \
  --batch-size 8 \
  --steer-strength -10.0 \
  --temperature 0.0 \
  --results-file logs/results/steered_humaneval_qwen2_5_run.jsonl |& tee logs/steered_humaneval_qwen2_5_run.log
