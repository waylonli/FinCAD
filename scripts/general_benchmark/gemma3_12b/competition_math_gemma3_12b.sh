#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/math-gemma3_12b.out
#$ -e ~/look-ahead-bias/logs/server-logs/math-gemma3_12b.err

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/competition_math/eval.py \
  --model-name ~/models/gemma-3-12b-it/ \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 1024 \
  --batch-size 16 \
  --temperature 0.0 \
  --results-file logs/results/competition_math_gemma3_12b_run.jsonl |& tee logs/competition_math_gemma3_12b_run.log
