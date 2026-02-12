import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tkz = load("deepseek-coder-1.3b-instruct-mlx/")

prompt = """<｜begin▁of▁sentence｜>
You are an AI code generator directly integrated into an IDE. Your responses
will be used by the IDE as-is, so they should only contain valid code.

### Instruction:

Give me Fibonacci in C

### Response:
"""
print(prompt, end="", flush=True)

cache = make_prompt_cache(model)
tokens = mlx.array(tkz.encode(prompt, cache=cache, add_special_tokens=False))[None]

while True:
    logits = model(tokens, cache=cache)
    nxt = mlx.argmax(logits[:, -1, :], axis=-1)
    if nxt.item() == tkz.eos_token_id: break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    tokens = nxt[None]
