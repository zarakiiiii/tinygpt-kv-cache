import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, start_tokens, block_size, max_new_tokens):
    """
    ❌ NON-CACHED GENERATION (SLOW)
    Recomputes attention over the full context every step.
    Time complexity: O(T^2)
    Used as baseline for benchmarking.
    """
    model.eval()

    tokens = start_tokens  # growing sequence of generated token IDs

    for _ in range(max_new_tokens):
        # crop context if too long
        # shape: (batch_size, <= block_size)
        # this means: take all batches and take last `block_size` tokens
        x_cond = tokens[:, -block_size:]

        # IMPORTANT:
        # - No past_kv passed here
        # - Model recomputes K,V for ALL tokens every step
        logits, _, _ = model(x_cond)

        # We select the last time step because that’s the model’s
        # prediction for the NEXT token
        logits = logits[:, -1, :]   # shape: (B, vocab_size)

        # GREEDY decoding (argmax)
        # Using sampling would introduce randomness → bad for benchmarking
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # concatenate → sequence grows by 1 token
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


@torch.no_grad()
def generate_with_cache(model, start_tokens, block_size, max_new_tokens):
    """
    ✅ CACHED GENERATION (FAST)
    Uses KV cache to avoid recomputing attention for past tokens.
    Time complexity: O(T)
    This mirrors how real GPT inference works.
    """
    model.eval()

    tokens = start_tokens  # growing sequence
    past_kv = None         # KV cache (list of per-layer (k, v))

    for _ in range(max_new_tokens):
        if past_kv is None:
            # First step:
            # No cache exists yet → need full context
            # this means: take all batches and take last `block_size` tokens
            x_cond = tokens[:, -block_size:]
        else:
            # Subsequent steps:
            # Only feed the LAST generated token
            # Past context is already stored in past_kv
            x_cond = tokens[:, -1:]

        # Model returns:
        # logits: (B, T, vocab_size)   T = no. of tokens in seq.
        # loss: None (during inference)
        # past_kv: updated cache for ALL layers
        logits, _, past_kv = model(x_cond, past_kv=past_kv)

        # Take prediction for next token
        logits = logits[:, -1, :]   # (B, vocab_size)

        # GREEDY decoding (important for fair comparison)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Append new token to sequence
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens
