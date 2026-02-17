import json
import random

# 1. Define your vocabulary
vars = ["CHEEZBURGER", "CEILING_CAT", "HAPPY_CAT", "KITTEN", "NOMZ"]
strings = ["OH HAI WORLD", "I CAN HAZ DATA", "HOVERCAT IZ REAL", "MEOW"]
numbers = [10, 42, 100, 7]

def generate_lolcode_sample():
    v1, v2 = random.sample(vars, 2)
    n1, n2 = random.sample(numbers, 2)
    s1 = random.choice(strings)
    
    templates = [
        {
            "user": f"Initialize {v1} to {n1} and print it.",
            "assistant": f"HAI 1.2\nI HAS A {v1} ITZ {n1}\nVISIBLE {v1}\nKTHXBYE"
        },
        {
            "user": f"Add {n1} and {n2} and store in {v1}.",
            "assistant": f"HAI 1.2\nI HAS A {v1} ITZ SUM OF {n1} AN {n2}\nVISIBLE {v1}\nKTHXBYE"
        },
        {
            "user": f"Print the string '{s1}' to the screen.",
            "assistant": f"HAI 1.2\nVISIBLE \"{s1}\"\nKTHXBYE"
        }
    ]
    return random.choice(templates)

# 2. Logic to generate, shuffle, and save
def save_datasets(total_count=550, train_ratio=0.9):
    # Generate everything into memory first
    all_samples = []
    for _ in range(total_count):
        sample = generate_lolcode_sample()
        entry = {
            "messages": [
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]}
            ]
        }
        all_samples.append(json.dumps(entry))

    # Shuffle to prevent any ordered-bias
    random.shuffle(all_samples)

    # Calculate the split point
    split_idx = int(total_count * train_ratio)
    train_data = all_samples[:split_idx]
    valid_data = all_samples[split_idx:]

    # Write both files
    with open("lolcode-dataset/train.jsonl", "w") as f:
        f.write("\n".join(train_data) + "\n")
    
    with open("lolcode-dataset/valid.jsonl", "w") as f:
        f.write("\n".join(valid_data) + "\n")

    print(f"Generated {len(train_data)} training samples and {len(valid_data)} validation samples.")

save_datasets()
