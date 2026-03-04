#!/bin/bash

#$ -l h_rt=4:00:00
#$ -l h_vmem=128G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/cad-single-stock-phi4_14b.out
#$ -e ~/look-ahead-bias/logs/server-logs/cad-single-stock-phi4_14b.err

TICKER=${1:-NVDA}

source ~/.bashrc
conda activate look-ahead-bias
export PYTHONPATH=PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name /data/weixianli/models/phi-4 \
  --use-chat-template \
  --attn-implementation flash_attention_2 \
  --ticker "${TICKER}" \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --rebalance-freq B \
  --decoding-mode cad \
  --cad-prior-mode optimized \
  --optimized-instruction results/discovery/phi-4.json \
  --use-calibrator \
  --temperature 0.0 \
  --max-new-tokens 256 \
  --results-file "results/backtest/single_${TICKER}_cad_phi4_14b.jsonl" \
  --summary-file "results/backtest/single_${TICKER}_cad_phi4_14b_summary.json" \
  --values-csv "results/backtest/single_${TICKER}_cad_phi4_14b_values.csv" \
  > "logs/cad_single_stock_${TICKER}_phi4_14b_outsample.log" 2>&1 &

echo "PID: $!"
echo "Log: tail -f logs/cad_single_stock_${TICKER}_phi4_14b_outsample.log"
