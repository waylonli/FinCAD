#!/bin/bash

#$ -l h_rt=10:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/fineval-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/fineval-phi4_14b.err



conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/fineval/eval.py \
  --model-name /data/weixianli/models/phi-4 \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --data-dir dataset/MMLU-Finance \
  --subset all \
  --max-new-tokens 64 \
  --batch-size 16 \
  --temperature 0.0 \
  --results-file logs/results/fineval_phi4_14b_run.jsonl |& tee logs/fineval_phi4_14b_run.log
