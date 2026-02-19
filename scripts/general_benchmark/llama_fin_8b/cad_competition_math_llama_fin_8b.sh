#!/bin/bash

#$ -l h_rt=6:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=1
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/cad-math-llama_fin_8b.out
#$ -e ~/look-ahead-bias/logs/server-logs/cad-math-llama_fin_8b.err

conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./

python benchmark/competition_math/eval.py \
  --model-name /data/weixianli/models/Llama-Fin-8b \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --dataset-cache-dir ./datasets \
  --max-new-tokens 1024 \
  --batch-size 16 \
  --temperature 0.0 \
  --decoding-mode cad \
  --cad-alpha 1.0 \
  --cad-top-p 1.0 \
  --cad-prior-mode question_only \
  --results-file logs/results/cad_competition_math_llama_fin_8b_run.jsonl |& tee logs/cad_competition_math_llama_fin_8b_run.log
