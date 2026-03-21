#!/bin/bash

#$ -l h_rt=2:00:00
#$ -l h_vmem=200G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/profile-gemma3_12b.out
#$ -e ~/look-ahead-bias/logs/server-logs/profile-gemma3_12b.err

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

python -m cad.discovery.profiler \
  --model-name ~/models/gemma-3-12b-it/ \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --attn-implementation flash_attention_2 \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --optimized-instruction results/discovery/gemma-3-12b-it.json \
  --alpha-target 3.0 \
  > logs/profile_gemma3_12b.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/profile_gemma3_12b.log"
