# TinyGPT with KV Cache

This project implements a minimal decoder-only Transformer (GPT-style)
from scratch in PyTorch, with correct KV cache support for efficient
autoregressive inference.

## Features
- Decoder-only Transformer
- Causal self-attention
- KV cache for O(T) inference
- Cache-aware positional embeddings
- Benchmark comparing cached vs non-cached decoding

## Why KV Cache?
Without KV cache, autoregressive decoding recomputes attention over all
past tokens (O(T²)). KV cache stores past keys and values, reducing
inference complexity to O(T).

## Benchmark (CPU, small model)
Cached decoding is faster than non-cached decoding, with speedup growing
as model size and sequence length increase.
