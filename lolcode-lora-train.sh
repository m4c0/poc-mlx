#!/bin/sh

python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train \
  --data ./lolcode \
  --iters 500 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --steps-per-report 10
