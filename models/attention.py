import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, block_size):
        super().__init__()

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = nn.Embedding(block_size, d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

    def forward(self, x, past_kv=None):
        B, T, C = x.shape

        # cache-aware positions
        if past_kv is None:
            start_pos = 0
        else:
            start_pos = past_kv[0].size(1) - T

        pos_ids = torch.arange(start_pos, start_pos + T, device=x.device)
        x = x + self.pos_emb(pos_ids)

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        new_kv = (k, v)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (C ** 0.5)

        T_total = k.size(1)
        # During training (no KV cache), we must apply causal masking
        # to prevent tokens from attending to future tokens.
        if past_kv is None:
            scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
            )

        # During inference with KV cache:
        # - past tokens are already guaranteed to be earlier in time
        # - current query only attends to past + current tokens
        # - no future tokens exist
        # → no masking needed


        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        return out, new_kv


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

    def forward(self, x, past_kv = None):
        B, T, C = x.shape

        qkv = self.qkv(x)                 # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)    #each (B, T, d_model) 
        
        #reshape into heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        #now (B, n_heads, T, head_dim)

        #------KV CACHE LOGIC------
        if past_kv is not None:
            past_k, past_v = past_kv      #(B, n_heads, T_past, head_dim)
            k = torch.cat([past_k, k], dim=2)    # concat on time dim
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)

        T_total = k.size(2)   #T_past + T

        #attention
        scores = q @ k.transpose(-2, -1)   # (B, n_heads, T, T_total)
        scores = scores / (self.head_dim ** 0.5)
        
        #causal mask
        # During training (no KV cache), we must apply causal masking
        # to prevent tokens from attending to future tokens.
        if past_kv is None:
            scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
            )

        # During inference with KV cache:
        # - past tokens are already guaranteed to be earlier in time
        # - current query only attends to past + current tokens
        # - no future tokens exist
        # → no masking needed


        weights = F.softmax(scores, dim=-1)
        out = weights @ v       # (B, n_heads, T, head_dim)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out =  self.proj(out)

        return out, new_kv
