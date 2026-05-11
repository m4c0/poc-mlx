import mlx.core as mx

adapters = mx.load("sentiment-adapter/adapters.safetensors")

# This converts from MLX standard:
# - F32 to BF16
# - The stride of each tensor (i.e. transposes them)
converted = {}
for name, arr in adapters.items():
    arr = arr.astype(mx.bfloat16)
    if ".lora_a" in name or ".lora_b" in name:
        arr = arr.T
    converted[name] = arr
mx.save_safetensors(
    "sentiment-adapter/adapters-peft.safetensors",
    converted
)
