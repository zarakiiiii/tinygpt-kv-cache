import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            #for above, eg: (B,T,d_model) -> (B,T,4*d_model)
            nn.GELU(),
            nn.Linear(4*d_model, d_model)  #(B,T,4*d_model) -> (B,T,d_model)
        )
    
    def forward(self, x):
        return self.net(x)