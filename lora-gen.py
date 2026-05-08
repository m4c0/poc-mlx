import json
import random

# https://huggingface.co/datasets/yuncongli/chat-sentiment-analysis

def conv(line):
    j = json.loads(line)
    inst = j['instruction']
    inpt = j['input']
    outp = j['output']
    text = f'<|im_start|>S<|im_end|>{inst}<|im_start|>U<|im_end|>{inpt}<|im_start|>A<|im_end|>{outp}<|endoftext|>'
    obj = { "text": text }
    return json.dumps(obj)

lines = []
with open("task_data.json") as f:
    lines = [conv(line) for line in f]

random.shuffle(lines)

count = len(lines)
split = int(count * 0.9)

train = lines[:split]
with open("sentiment/train.jsonl", "w") as f:
    f.write("\n".join(train) + "\n")

valid = lines[split:]
with open("sentiment/valid.jsonl", "w") as f:
    f.write("\n".join(valid) + "\n")
