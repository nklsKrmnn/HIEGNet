import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, _) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

