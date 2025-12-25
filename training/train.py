print(">>> train.py started")

import torch

from models.gpt import TinyGPT
from training.data import get_batch
from config import (
    vocab_size,
    d_model,
    n_heads,
    block_size,
    n_layers,
    batch_size,
    epochs,
    lr,
)

# ✅ sanity check goes HERE
assert epochs > 0, "epochs must be > 0"

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    block_size=block_size,
    n_layers=n_layers
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch(batch_size, block_size, vocab_size)

    logits, loss, _ = model(xb, yb)

    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute new gradients
    optimizer.step()        # update weights

    if step % 20 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
        # loss.item() converts 0-dim tensor to python number.
        # loss is a tensor. you cant print/format it until you convert it.
        # .4f formats as a float, and shows 4 decimal places
