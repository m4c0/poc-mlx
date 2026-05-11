#!/bin/sh
set -e

# https://huggingface.co/datasets/yuncongli/chat-sentiment-analysis

python lora-gen.py

python lora-train.py

# F32 to BF16, MLX to PEFT
python lora-conv.py
