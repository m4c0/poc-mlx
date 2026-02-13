import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
cache = [KVCache() for _ in range(len(model.layers))]

prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: bash

Constraint: Output only raw code one-liners. No explanatory text.

Example: if user asks to sort file X, your answer should be

sort X

No backticks or any markdown.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given a log file named "my.log", containing lines in the following format:

2015-02-30 04:50:10 [<level>] <thread-id> <message>

Where "level" can be "I" for "Info", "D" for "Debug" or "E" for "Error".

How do I get a list of errors sorted by thread id?

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
enc = tkz.encode(prompt, add_special_tokens=False)
print(enc);

tokens = mlx.array(enc)[None]
print(tkz.decode(enc), end="", flush=True)

while True:
    logits = model(tokens, cache=cache)
    nxt = mlx.argmax(logits[:, -1, :], axis=-1)
    if nxt.item() == tkz.eos_token_id: break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    tokens = nxt[None]
