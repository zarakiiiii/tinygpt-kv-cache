import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        
        #x → LayerNorm → Multi-Head Attention → Add (residual)
        #→ LayerNorm → FeedForward (MLP)    → Add (residual)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        #Why TWO LayerNorms?
        #One before attention
        #One before FFN

        self.attn = MultiHeadAttention(d_model, n_heads, block_size)
        self.ffn = FeedForward(d_model)

    def forward(self, x, past_kv = None):
       """
       past_kv: tuple(k, v) or None
       """
       attn_out, new_kv = self.attn(self.ln1(x), past_kv)
       x = x + attn_out
       x = x + self.ffn(self.ln2(x))
       return x, new_kv
                