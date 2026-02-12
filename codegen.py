import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

msgs = [{
  "role": "system",
  "content": "You are a raw C code generator. Write concise, idiomatic code. Output ONLY the code. DO NOT provide explanations."
}, {
  "role": "user",
  "content": "Write a function to calculate the Fibonacci sequence."
}]
prompt = tkz.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
prompt += "[code]"
#print("int", end="", flush=True)

tokens = mlx.array(tkz.encode(prompt))[None]

cache = [KVCache() for _ in range(len(model.layers))]

logits = model(tokens, cache=cache)
for _ in range(1000):
    nxt = mlx.argmax(logits[:, -1, :], axis=-1)

    if nxt.item() == tkz.eos_token_id:
        break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    logits = model(nxt[None], cache=cache)

