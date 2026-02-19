#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/cad-mmlu-pro-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/cad-mmlu-pro-phi4_14b.err



conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/mmlu_pro/eval.py \
  --model-name /data/weixianli/models/phi-4 \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 2048 \
  --batch-size 16 \
  --temperature 0.0 \
  --decoding-mode cad \
  --cad-alpha 1.0 \
  --cad-top-p 1.0 \
  --cad-prior-mode question_only \
  --results-file logs/results/cad_mmlu_pro_phi4_14b_run.jsonl |& tee logs/cad_mmlu_pro_phi4_14b_run.log
