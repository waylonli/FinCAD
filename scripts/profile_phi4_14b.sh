#!/bin/bash

#$ -l h_rt=2:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/profile-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/profile-phi4_14b.err

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

python -m cad.discovery.profiler \
  --model-name /data/weixianli/models/phi-4 \
  --use-chat-template \
  --attn-implementation flash_attention_2 \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --optimized-instruction results/discovery/phi-4.json \
  --alpha-target 3.0 \
  > logs/profile_phi4_14b.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/profile_phi4_14b.log"
