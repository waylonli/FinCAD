#!/bin/bash

#$ -l h_rt=8:00:00
#$ -l h_vmem=200G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/baseline-single-stock-gemma3_12b.out
#$ -e ~/look-ahead-bias/logs/server-logs/baseline-single-stock-gemma3_12b.err

TICKER=${1:-NVDA}

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name ~/models/gemma-3-12b-it/ \
  --use-chat-template \
  --model-cache-dir ../pretrained_models \
  --attn-implementation flash_attention_2 \
  --ticker "${TICKER}" \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --start-date 2010-01-01 \
  --end-date 2020-01-01 \
  --rebalance-freq B \
  --decoding-mode baseline \
  --temperature 0.0 \
  --max-new-tokens 256 \
  --results-file "results/backtest/single_${TICKER}_baseline_gemma3_12b.jsonl" \
  --summary-file "results/backtest/single_${TICKER}_baseline_gemma3_12b_summary.json" \
  --values-csv "results/backtest/single_${TICKER}_baseline_gemma3_12b_values.csv" \
  > "logs/baseline_single_stock_${TICKER}_gemma3_12b.log" 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/baseline_single_stock_${TICKER}_gemma3_12b.log"
