from torch import nn
import torch

class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()

        self.lin1 = nn.Linear(insize + hsize, hsize)
        self.lin2 = nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):
        b, t, e = x.size()

        # Initialize hidden with zeros if not given
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)

        outs = []

        for i in range(t):
            # Concat previous hidden layer with input
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            
            # First layer is our new hidden
            hidden = self.lin1(inp)
            # Output layer
            out = self.lin2(hidden)

            outs.append(out[:, None, :])

        return torch.cat(outs, dim=1), hidden
