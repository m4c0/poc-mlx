import json
import random

# https://huggingface.co/datasets/yuncongli/chat-sentiment-analysis

def conv(line):
    j = json.loads(line)
    msgs = [
        { "role": "system",    "content": j['instruction'] },
        { "role": "user",      "content": j['input'] },
        { "role": "assistant", "content": j['output'] },
    ]
    return json.dumps({ "messages": msgs })

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
