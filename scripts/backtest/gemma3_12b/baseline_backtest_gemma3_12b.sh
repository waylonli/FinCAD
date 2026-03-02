#!/bin/bash

#$ -l h_rt=24:00:00
#$ -l h_vmem=200G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/baseline-backtest-gemma3_12b.out
#$ -e ~/look-ahead-bias/logs/server-logs/baseline-backtest-gemma3_12b.err

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

python -m benchmark.backtest.q_scores_eval.eval \
  --model-name ~/models/gemma-3-12b-it/ \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --attn-implementation flash_attention_2 \
  --score-mode on_demand \
  --decoding-mode baseline \
  --temperature 0.0 \
  --max-new-tokens 512 \
  --max-filing-chars 60000 \
  --chunk-size 4096 \
  --chunk-overlap 128 \
  --symbols all \
  --start-year 2014 \
  --end-year 2024 \
  --top-quantile 0.05 \
  --results-file results/backtest/filings_baseline_gemma3_12b.jsonl \
  --summary-file results/backtest/summary_baseline_gemma3_12b.json \
  > logs/baseline_backtest_gemma3_12b.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/baseline_backtest_gemma3_12b.log"
