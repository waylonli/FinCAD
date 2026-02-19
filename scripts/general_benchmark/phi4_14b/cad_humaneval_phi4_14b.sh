#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/cad-humaneval-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/cad-humaneval-phi4_14b.err


conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/humaneval/eval.py \
  --model-name /data/weixianli/models/phi-4 \
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
  --results-file logs/results/cad_humaneval_phi4_14b_run.jsonl |& tee logs/cad_humaneval_phi4_14b_run.log
