#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o /exports/eddie/scratch/s1891340/look-ahead-bias/logs/server-logs/cad-humaneval-qwen2_5.out
#$ -e /exports/eddie/scratch/s1891340/look-ahead-bias/logs/server-logs/cad-humaneval-qwen2_5.err

cd /exports/eddie/scratch/s1891340/look-ahead-bias
source ~/.bashrc
source /exports/csce/eddie/inf/groups/FinComputing/waylon/venv/look-ahead/bin/activate
export PYTHONPATH=PYTHONPATH:./

uv run --active --no-sync python benchmark/humaneval/eval.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 256 \
  --batch-size 8 \
  --temperature 0.0 \
  --decoding-mode cad \
  --cad-alpha 1.0 \
  --cad-top-p 1.0 \
  --cad-prior-mode question_only \
  --results-file logs/results/cad_humaneval_qwen2_5_run.jsonl |& tee logs/cad_humaneval_qwen2_5_run.log
