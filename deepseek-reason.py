import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tkz = load("mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-MLX-Q8")

prompt = """<｜begin▁of▁sentence｜>
Be short and concise in your answers.
Code suggestions must be in C, always.
<｜User｜>
How to add a recursion limit to this function?

int fib(int n) {
    if (n == 1) return 1;
    if (n == 2) return 1;
    return fib(n - 1) + fib(n - 2);
}
<｜Assistant｜>
"""

cache = make_prompt_cache(model)
pp = tkz.encode(prompt, cache=cache, add_special_tokens=False)
print(tkz.decode(pp), end="", flush=True)
tokens = mlx.array(pp)[None]

while True:
    logits = model(tokens, cache=cache)
    nxt = mlx.argmax(logits[:, -1, :], axis=-1)
    if nxt.item() == tkz.eos_token_id: break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    tokens = nxt[None]
