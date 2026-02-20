#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/cad-humaneval-gemma3_12b.out
#$ -e ~/look-ahead-bias/logs/server-logs/cad-humaneval-gemma3_12b.err

conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/humaneval/eval.py \
  --model-name ~/models/gemma-3-12b-it/ \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 256 \
  --batch-size 8 \
  --temperature 0.0 \
  --decoding-mode cad \
  --cad-alpha 1.0 \
  --cad-top-p 1.0 \
  --cad-prior-mode recall \
  --results-file logs/results/cad_humaneval_gemma3_12b_run.jsonl |& tee logs/cad_humaneval_gemma3_12b_run.log
