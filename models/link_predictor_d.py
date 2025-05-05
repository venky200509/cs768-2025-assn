import torch
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.0):
        super(LinkPredictor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Same structure, just better initialization
        self.src_transform = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.dst_transform = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Keep same interaction structure
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Same final layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, 1))
        
        # Added temperature parameter for better calibration
        self.temperature = torch.nn.Parameter(torch.ones(1))
        
        # Better initialization
        self._reset_parameters()

    def _reset_parameters(self):
        """Better weight initialization"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x_i, x_j):
        # Transform source and target embeddings separately
        src_emb = self.src_transform(x_i)  # [batch_size, hidden_channels]
        dst_emb = self.dst_transform(x_j)  # [batch_size, hidden_channels]
        
        # SIMPLE CHANGE 1: Add source-destination difference
        # This helps the model capture directionality better
        difference = src_emb - dst_emb  # [batch_size, hidden_channels]
        
        # Original similarity
        similarity = src_emb * dst_emb  # [batch_size, hidden_channels]
        
        # SIMPLE CHANGE 2: Combine difference info with similarity
        # This improves directional sensitivity without changing architecture
        enhanced_sim = similarity + 0.1 * torch.abs(difference)
        
        # Keep original concatenation approach
        interaction = torch.cat([src_emb, dst_emb], dim=-1)  # [batch_size, 2*hidden_channels]
        interaction = self.interaction(interaction)  # [batch_size, hidden_channels]
        
        # SIMPLE CHANGE 3: Better feature combination 
        x = enhanced_sim + interaction  # [batch_size, hidden_channels]
        
        # Apply final layers (unchanged)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        
        # SIMPLE CHANGE 4: Better scaled sigmoid
        # The temperature parameter helps differentiate scores
        return torch.sigmoid(x / (self.temperature + 1e-10))