#!/bin/bash

#$ -l h_rt=10:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/gsm8k-gemma3_12b.out
#$ -e ~/look-ahead-bias/logs/server-logs/gsm8k-gemma3_12b.err

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/gsm8k/eval.py \
  --model-name ~/models/gemma-3-12b-it/ \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 512 \
  --batch-size 16 \
  --temperature 0.0 \
  --results-file logs/results/gsm8k_gemma3_12b_run.jsonl |& tee logs/gsm8k_gemma3_12b_run.log
