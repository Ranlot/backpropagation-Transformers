import torch
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha = 16):
        super().__init__()
        self.D = torch.nn.Parameter(torch.empty(in_dim, rank))
        self.U = torch.nn.Parameter(torch.empty(rank, out_dim))
        torch.nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))  
        torch.nn.init.kaiming_uniform_(self.D, a=math.sqrt(5))  
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.D @ self.U)
        return x


