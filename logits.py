import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
tokens = mlx.array(tkz.encode("The capital of Paraiba is"))[None]

cache = [KVCache() for _ in range(len(model.layers))]

logits = model(tokens, cache=cache)
nxt = mlx.argmax(logits[:, -1, :], axis=-1)
print(tkz.decode(nxt.tolist()))
tokens = mlx.concatenate([tokens, nxt[None]], axis=1)

logits = model(tokens, cache=cache)
nxt = mlx.argmax(logits[:, -1, :], axis=-1)
print(tkz.decode(nxt.tolist()))
tokens = mlx.concatenate([tokens, nxt[None]], axis=1)

logits = model(tokens, cache=cache)
nxt = mlx.argmax(logits[:, -1, :], axis=-1)
print(tkz.decode(nxt.tolist()))
