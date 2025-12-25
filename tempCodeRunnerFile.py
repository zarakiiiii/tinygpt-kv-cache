def forward(self, x):
    B, T = x.shape  # x is token ID's shape is (B,T)
    tok_emb = self.token_embedding(x)  # shape (B, T, d_model)
    pos = torch.arange(0, T, device=x.device)
    pos_emb = self.position_embedding(pos)  # shape (T, d_model)
    x = tok_emb + pos_emb

    for block in self.blocks:
        x = block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    loss = None  # useful during inference when no targets provided
    if targets is not None:
        # flatten for cross entropy
        # f.cross_entropy expects : i/p = (N,C) & o/p = (N) where N = no. of predictions, C = no. of classes
        # But language modeling predicts one token per position, so total predictions: N = B × T
        logits = logits.view(B * T, -1)  # (B, T, vocab_size) -> (B*T, vocab_size)
        targets = targets.view(B * T)  # (B, T) -> (B*T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss
