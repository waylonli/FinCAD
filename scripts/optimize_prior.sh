#!/bin/bash

#$ -l h_rt=8:00:00
#$ -l h_vmem=200G
#$ -q gpu
#$ -l gpu=2
#$ -P inf_fincomputing
#$ -o ~/look-ahead-bias/logs/server-logs/optimize-prior.out
#$ -e ~/look-ahead-bias/logs/server-logs/optimize-prior.err

eval "$(conda shell.bash hook)"
conda activate look-ahead-bias
export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

PRICE_CSV=dataset/backtest-data/price/price_data.csv
OUTPUT_DIR=results/discovery
# Toggle: "transformers" (in-process, no server) or "vllm" (auto-start server)
BACKEND="${BACKEND:-transformers}"
VLLM_PORT=8234
COMMON_ARGS="--optimizer MIPROv2 --num-candidates 25 --num-trials 50 \
  --forward-days 63 --max-examples 200 --min-abs-return 0.05 \
  --date-range-start 2005-01-01 --date-range-end 2015-01-01"

run_optimize() {
  local model_path=$1
  local model_label=$2

  if [ "${BACKEND}" = "vllm" ]; then
    echo "=== Starting vLLM server for ${model_label} ==="
    python -m vllm.entrypoints.openai.api_server \
      --model "${model_path}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size 1 \
      --dtype auto &
    VLLM_PID=$!

    echo "Waiting for vLLM server (PID ${VLLM_PID}) ..."
    for i in $(seq 1 120); do
      if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
      fi
      sleep 1
    done

    echo "=== Optimizing prior for ${model_label} (vLLM) ==="
    python -m cad.discovery \
      --model-name "${model_path}" \
      --server-url "http://localhost:${VLLM_PORT}/v1" \
      --price-csv "${PRICE_CSV}" \
      --output-dir "${OUTPUT_DIR}" \
      ${COMMON_ARGS}

    echo "=== Stopping vLLM server ==="
    kill "${VLLM_PID}" 2>/dev/null
    wait "${VLLM_PID}" 2>/dev/null
  else
    echo "=== Optimizing prior for ${model_label} (transformers) ==="
    python -m cad.discovery \
      --model-name "${model_path}" \
      --price-csv "${PRICE_CSV}" \
      --output-dir "${OUTPUT_DIR}" \
      ${COMMON_ARGS}
  fi
}

run_optimize /data/weixianli/models/phi-4 "phi-4 (14B)"
run_optimize ~/models/gemma-3-12b-it "gemma-3-12b-it"
