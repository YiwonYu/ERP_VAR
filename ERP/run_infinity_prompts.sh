#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/ERP/outputs"
mkdir -p "$OUTPUT_DIR"

PROMPT_FILES=(
  "$ROOT_DIR/prompts_100_v1.txt"
  "$ROOT_DIR/prompts_panorama.txt"
)

declare -a PROMPTS
declare -a OUTPUTS

for prompt_file in "${PROMPT_FILES[@]}"; do
  if [[ ! -f "$prompt_file" ]]; then
    echo "Prompt file not found: $prompt_file" >&2
    exit 1
  fi

  base_name="$(basename "$prompt_file")"
  base_name="${base_name%.*}"
  idx=0

  while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    if [[ -z "${prompt// }" ]]; then
      continue
    fi
    idx=$((idx + 1))
    PROMPTS+=("$prompt")
    OUTPUTS+=("$OUTPUT_DIR/${base_name}_${idx}.jpg")
  done < "$prompt_file"
done

run_shard() {
  local gpu_id="$1"
  local shard_id="$2"
  local shard_total="$3"
  local total="${#PROMPTS[@]}"
  local i
  for ((i = shard_id; i < total; i += shard_total)); do
    CUDA_VISIBLE_DEVICES="$gpu_id" \
      python "$ROOT_DIR/Infinity/inference.py" \
      --prompt "${PROMPTS[$i]}" \
      --save_file "${OUTPUTS[$i]}"
  done
}

run_shard 0 0 3 &
run_shard 1 1 3 &
run_shard 2 2 3 &
wait
