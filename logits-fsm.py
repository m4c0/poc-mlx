import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
import os

# This example does not work as intended

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
cache = [KVCache() for _ in range(len(model.layers))]

prompt = """
Given a blergh starts with ACG and each slurp is either CCA or GGA or TAC.

Blergh with three slurps would be:"""

slurps = [tkz.encode(c, add_special_tokens=False)[0] for c in ["A", "C", "G", "T"]]

voc_size = len(model.model.embed_tokens.biases)
mask = mlx.full((voc_size, ), -float("inf"))
mask[mlx.array(slurps)] = 0.0

enc = tkz.encode(prompt, add_special_tokens=False)
tokens = mlx.array(enc)[None]
print(tkz.decode(enc))

for i in range(1000):
    logits = model(tokens, cache=cache)
    last = logits[:, -1, :]

    if i == 0:
        nxt = mlx.array([tkz.encode("A")[0]])
    elif i == 1:
        nxt = mlx.array([tkz.encode("C")[0]])
    elif i == 2:
        nxt = mlx.array([tkz.encode("G")[0]])
    else:
        nxt = mlx.argmax(last + mask, axis=-1)

    if nxt.item() == tkz.eos_token_id: break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    tokens = nxt[None]
