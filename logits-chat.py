import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
cache = [KVCache() for _ in range(len(model.layers))]

prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your answers should be as tiny as possible.

<|eot_id|><|start_header_id|>user<|end_header_id|>

What's the capital of Paraiba?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
print(prompt, end="", flush=True)

tokens = mlx.array(tkz.encode(prompt))[None]

while True:
    logits = model(tokens, cache=cache)
    nxt = mlx.argmax(logits[:, -1, :], axis=-1)
    if nxt.item() == tkz.eos_token_id: break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    tokens = nxt[None]
