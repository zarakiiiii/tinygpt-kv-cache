import torch

def get_batch(batch_size, block_size, vocab_size):
     #batch_size = no. of training examples processed together in 1 forward/backward pass
     x = torch.randint(0, vocab_size, (batch_size, block_size))  #torch.randint(low, high, size)  
     y = torch.randint(0, vocab_size, (batch_size, block_size))
     #x = (B,T) & y = (B,T) , x = input tokens, y = target token
     #x = what the model sees; y = what the model should predict.
     return x, y