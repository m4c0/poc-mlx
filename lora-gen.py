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
split1 = int(count * 0.8)
split2 = int(count * 0.9)

train = lines[:split1]
with open("sentiment/train.jsonl", "w") as f:
    f.write("\n".join(train) + "\n")

valid = lines[split1:split2]
with open("sentiment/valid.jsonl", "w") as f:
    f.write("\n".join(valid) + "\n")

tests = lines[split2:]
with open("sentiment/test.jsonl", "w") as f:
    f.write("\n".join(tests) + "\n")
