from mlx_lm import load, generate

model, tokenz = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

msgs = [{ "role": "system", "content": "your answers should be short and suitable for AuDHD readers" },
        { "role": "user", "content": "whats the colour of the sky?"}]
prompt = tokenz.apply_chat_template(msgs, add_generation_prompt=True)

print(generate(model, tokenz, prompt))
