#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/humaneval-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/humaneval-phi4_14b.err



conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/humaneval/eval.py \
  --model-name /data/weixianli/models/phi-4 \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 2048 \
  --batch-size 8 \
  --temperature 0.0 \
  --results-file logs/results/humaneval_phi4_14b_run.jsonl |& tee logs/humaneval_phi4_14b_run.log
