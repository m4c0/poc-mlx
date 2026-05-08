#!/bin/sh

# https://huggingface.co/datasets/yuncongli/chat-sentiment-analysis

time python -m mlx_lm lora \
  --model HuggingFaceTB/SmolLM2-135M \
  --train \
  --data ./sentiment \
  --adapter-path ./sentiment-adapter \
  --iters 500 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --steps-per-report 10
