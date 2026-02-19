#!/bin/bash

#$ -l h_rt=24:00:00
#$ -l h_vmem=200G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/baseline-backtest-qwen2_5_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/baseline-backtest-qwen2_5_14b.err

conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

python -m benchmark.backtest.q_scores_eval.eval \
  --model-name /data/weixianli/models/Qwen2.5-14B-Instruct \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --attn-implementation flash_attention_2 \
  --score-mode on_demand \
  --decoding-mode baseline \
  --temperature 0.0 \
  --max-new-tokens 512 \
  --max-filing-chars 60000 \
  --chunk-size 8192 \
  --chunk-overlap 256 \
  --symbols all \
  --start-year 2014 \
  --end-year 2024 \
  --top-quantile 0.05 \
  --results-file results/backtest/filings_baseline_qwen2_5_14b.jsonl \
  --summary-file results/backtest/summary_baseline_qwen2_5_14b.json \
  > logs/baseline_backtest_qwen2_5_14b.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/baseline_backtest_qwen2_5_14b.log"
