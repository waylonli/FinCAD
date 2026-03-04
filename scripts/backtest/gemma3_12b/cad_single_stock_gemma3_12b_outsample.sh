#!/bin/bash

#$ -l h_rt=4:00:00
#$ -l h_vmem=200G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/cad-single-stock-gemma3_12b-outsample.out
#$ -e ~/look-ahead-bias/logs/server-logs/cad-single-stock-gemma3_12b-outsample.err

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
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --rebalance-freq B \
  --decoding-mode cad \
  --cad-prior-mode optimized \
  --optimized-instruction results/discovery/gemma-3-12b-it.json \
  --use-calibrator \
  --temperature 0.0 \
  --max-new-tokens 256 \
  --results-file "results/backtest/single_${TICKER}_cad_gemma3_12b_outsample.jsonl" \
  --summary-file "results/backtest/single_${TICKER}_cad_gemma3_12b_outsample_summary.json" \
  --values-csv "results/backtest/single_${TICKER}_cad_gemma3_12b_outsample_values.csv" \
  > "logs/cad_single_stock_${TICKER}_gemma3_12b_outsample.log" 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/cad_single_stock_${TICKER}_gemma3_12b_outsample.log"
