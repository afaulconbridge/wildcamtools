#!/bin/bash

# shellcheck disable=SC1091
set -a && source pipelines/.env && set +a

uv run wildcamtools ai run-evaluate --help


#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_prompt.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/prompt.json 2>&1 | tee pipelines/output/output_prompt.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_verify_high.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/verify_high.json 2>&1 | tee pipelines/output/output_verify_high.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_verify_med.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/verify_med.json 2>&1 | tee pipelines/output/output_verify_med.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_motion.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/motion.json 2>&1 | tee pipelines/output/output_motion.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_ssim.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/ssim.json 2>&1 | tee pipelines/output/output_ssim.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_crop.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/crop.json 2>&1 | tee pipelines/output/output_crop.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/output.json 2>&1 | tee pipelines/output/output.log
#uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_5x1280x720.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/5x1280x720.json 2>&1 | tee pipelines/output/output_5x1280x720.log
uv run wildcamtools ai run-evaluate -w 4 pipelines/pipeline_config_contrast.json pipelines/comparison_config.json wildlife/2023/labels.jsonl --output pipelines/output/output_contrast.json 2>&1 | tee pipelines/output/output_contrast.log
