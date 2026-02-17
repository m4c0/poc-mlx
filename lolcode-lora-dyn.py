from mlx_lm import load, generate
from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

print()
print("================== Without Adapters")
print(generate(model, tkz, "Write a loop that prints HAI 3 in LOLCODE times"))

load_adapters(model, "./lolcode-adapter")

print("================== With Adapters")
print(generate(model, tkz, "Write a loop that prints HAI 3 in LOLCODE times"))

remove_lora_layers(model)

print("================== Without Adapters")
print(generate(model, tkz, "Write a loop that prints HAI 3 in LOLCODE times"))

