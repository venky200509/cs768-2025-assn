import torch
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LinkPredictor, self).__init__()
        
        
        # Create linear layers
        self.lin1 = torch.nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        
        # Apply layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        # Apply sigmoid
        return torch.sigmoid(x) 