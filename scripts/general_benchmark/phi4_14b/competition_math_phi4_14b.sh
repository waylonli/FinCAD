#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/math-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/math-phi4_14b.err



conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/competition_math/eval.py \
  --model-name /data/weixianli/models/phi-4 \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 1024 \
  --batch-size 16 \
  --temperature 0.0 \
  --results-file logs/results/competition_math_phi4_14b_run.jsonl |& tee logs/competition_math_phi4_14b_run.log
