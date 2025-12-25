import time
import torch

from models.gpt import TinyGPT
from inference.generate import generate
from inference.generate import generate_with_cache
from config import (
    vocab_size,
    d_model,
    n_heads,
    block_size,
    n_layers,
)

# ------------------ setup ------------------
device = "cpu"  # keep CPU for clear comparison

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    block_size=block_size,
    n_layers=n_layers,
).to(device)

model.eval()

start_tokens = torch.randint(0, vocab_size, (1, 1), device=device)
#torch.randint(low, high, size)  high is exclusive
num_new_tokens = 200

# sanity check only within context window
N = block_size - 1

out1 = generate(model, start_tokens.clone(), block_size, N)
out2 = generate_with_cache(model, start_tokens.clone(), block_size, N)

assert torch.allclose(out1.float(), out2.float(), atol=1e-5), \
    "Mismatch before context window!"
print("✅ Sanity check passed (numerical tolerance)")


import time

@torch.no_grad()
def benchmark_no_cache():
    start = time.time()
    generate(
        model,
        start_tokens.clone(),
        block_size,
        num_new_tokens,
    )
    return time.time() - start


@torch.no_grad()
def benchmark_with_cache():
    start = time.time()
    generate_with_cache(
        model,
        start_tokens.clone(),
        block_size,
        num_new_tokens,
    )
    return time.time() - start


# warmup (important for fair timing)
benchmark_no_cache()
benchmark_with_cache()

# actual timing
t_no_cache = benchmark_no_cache()
t_cache = benchmark_with_cache()

print(f"Non-cached generation: {t_no_cache:.4f} seconds")
print(f"Cached generation:     {t_cache:.4f} seconds")
print(f"Speedup:               {t_no_cache / t_cache:.2f}x")





