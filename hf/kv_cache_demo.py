print(">>> kv_cache_demo.py STARTED")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cpu"

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2").to(device)
model.eval()

text = "Hello, my name is"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

past_key_values = None
generated_ids = input_ids.clone()

print("Generating tokens using KV cache:")

for step in range(30):  # increase for longer paragraph
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )

    logits = outputs.logits
    past_key_values = outputs.past_key_values

    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    input_ids = next_token

print("\n=== Generated paragraph ===\n")
print(tokenizer.decode(generated_ids[0]))
