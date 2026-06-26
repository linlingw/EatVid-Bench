#!/bin/bash
# EatVid-Bench Evaluation Example Script
# This script demonstrates how to run evaluation on the benchmark

set -e

# Configuration
MODEL="qwen2_5_vl_7b"
SPLIT="test"
OUTPUT_DIR="./results"
CONFIG_PATH="configs/benchmark_config.yaml"

# Optional: Enable GPT Judge
# export GPT_JUDGE_API_KEY="your-api-key"

echo "=== EatVid-Bench Evaluation ==="
echo "Model: $MODEL"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_DIR"
echo ""

# Run evaluation
python -m eatvid_benchmark.run_qa_baseline \
  --model "$MODEL" \
  --split "$SPLIT" \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG_PATH"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
cat "$OUTPUT_DIR/qa_evaluation/$SPLIT/qa_metrics.json"
