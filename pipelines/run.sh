#!/bin/bash

# shellcheck disable=SC1091
set -a && source pipelines/.env && set +a

uv run wildcamtools ai run-evaluate --help


uv run wildcamtools ai run-evaluate pipelines/pipeline_config_prompt.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/prompt.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config_verify_high.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/verify_high.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config_verify_med.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/verify_med.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config_motion.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/motion.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config_ssim.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/ssim.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config.json pipelines/comparison_config_crop.json wildlife/2023/labels.jsonl --output pipelines/output/crop.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/output.json
uv run wildcamtools ai run-evaluate pipelines/pipeline_config.json pipelines/comparison_config_5x1280x720.json wildlife/2023/labels.jsonl --output pipelines/output/5x1280x720.json
