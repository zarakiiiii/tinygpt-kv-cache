import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block import TransformerBlock

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, block_size, n_layers):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets = None, past_kv=None):
        """
        x: (B, T)
        past_kv: list of length n_layers, each element is (k, v) or None
        """

        tok_emb = self.token_embedding(x)   #shape (B, T, d_model)

        #----cache-aware positional embeddings------
        B, T = x.shape

        if past_kv is None:
        # training or first inference step
            pos = torch.arange(0, T, device=x.device)

        else:
            # past_kv[0][0] shape: (B, n_heads, T_past, head_dim)
            T_past = past_kv[0][0].size(2)

            # Absolute positions for new tokens
            pos = torch.arange(T_past, T_past + T, device=x.device)

            # IMPORTANT:
            # Clamp ONLY after exceeding context window
            pos = pos.clamp(max=self.position_embedding.num_embeddings - 1)

        x = tok_emb + self.position_embedding(pos)
        #------------------------------------------------

        new_past_kv = []

        for i, block in enumerate(self.blocks):
            layer_past = None if past_kv is None else past_kv[i]
            x, layer_kv = block(x, layer_past)
            new_past_kv.append(layer_kv)
        
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        loss = None #useful during inference when no targets provided
        if targets is not None:
                #flatten for cross entropy
                # f.cross_entropy expects : i/p = (N,C) & o/p = (N) where N = no. of predictions, C = no. of classes
                # But language modeling predicts one token per position, so total predictions: N = B × T
                logits_flat = logits.view(B * T, -1) # (B, T, vocab_size) -> (B*T, vocab_size)
                targets_flat = targets.view(B * T) #(B, T) -> (B*T)
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_past_kv