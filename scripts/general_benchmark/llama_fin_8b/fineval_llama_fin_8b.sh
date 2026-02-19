#!/bin/bash

#$ -l h_rt=10:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/fineval-llama_fin_8b.out
#$ -e ~/look-ahead-bias/logs/server-logs/fineval-llama_fin_8b.err



conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/fineval/eval.py \
  --model-name /data/weixianli/models/Llama-Fin-8b \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --data-dir dataset/MMLU-Finance \
  --subset all \
  --max-new-tokens 64 \
  --batch-size 16 \
  --temperature 0.0 \
  --results-file logs/results/fineval_llama_fin_8b_run.jsonl |& tee logs/fineval_llama_fin_8b_run.log
