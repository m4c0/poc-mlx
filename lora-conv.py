import mlx.core as mx
adapters = mx.load("sentiment-adapter/adapters.safetensors")
adapters_bf16 = {k: v.astype(mx.bfloat16) for k, v in adapters.items()}
mx.save_safetensors("sentiment-adapter/adapters.safetensors", adapters_bf16)
