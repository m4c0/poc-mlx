import types
import yaml
from mlx_lm.tuner import datasets
from mlx_lm.utils import load
from mlx_lm import lora

params = dict(lora.CONFIG_DEFAULTS)

with open("lora-config.yaml") as f:
    for k, v in yaml.safe_load(f).items():
        params[k] = v

args = types.SimpleNamespace(**params)

model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

train_set, valid_set, test_set = datasets.load_dataset(args, tokenizer)
lora.train_model(args, model, train_set, valid_set)
lora.evaluate_model(args, model, test_set)

