#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/mmlu-pro-llama_fin_8b.out
#$ -e ~/look-ahead-bias/logs/server-logs/mmlu-pro-llama_fin_8b.err



conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/mmlu_pro/eval.py \
  --model-name /data/weixianli/models/Llama-Fin-8b \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 2048 \
  --batch-size 16 \
  --temperature 0.0 \
  --split validation \
  --results-file logs/results/mmlu_pro_llama_fin_8b_run.jsonl |& tee logs/mmlu_pro_llama_fin_8b_run.log

# TODO Now it is set to validation, change to test for final evaluation
