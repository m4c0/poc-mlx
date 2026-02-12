import mlx.core as mlx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

model, tkz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
cache = [KVCache() for _ in range(len(model.layers))]

bos_id = tkz.bos_token_id
eos_id = tkz.eos_token_id
user_start = tkz.convert_tokens_to_ids("<|start_header_id|>")
user_end   = tkz.convert_tokens_to_ids("<|end_header_id|>")
eot_id     = tkz.convert_tokens_to_ids("<|eot_id|>")

prompt = "The capital of Paraiba is"
# This is incorrect, but it put the LLM in a "hybrid mode", where it outputs a
# description of Paraiba, followed by some self-review of the text - most
# likely from its RLHF (Reinforcement Learning from Human Feedback)
enc = [128010] + tkz.encode(prompt, add_special_tokens=False)

tokens = mlx.array(enc)[None]
print(tkz.decode(enc), end="", flush=True)

while True:
    logits = model(tokens, cache=cache)
    nxt = mlx.argmax(logits[:, -1, :], axis=-1)
    if nxt.item() == tkz.eos_token_id: break

    print(tkz.decode(nxt.tolist()), end="", flush=True)
    tokens = nxt[None]
