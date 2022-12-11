import torch
from torch import nn
from utils import DEVICE, GlobalMaxPool


class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300, activation=nn.ReLU):
        super().__init__()

        self.lin1 = nn.Sequential(nn.Linear(insize + hsize, hsize), activation())
        self.lin2 = nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):
        b, t, e = x.size()

        # Initialize hidden with zeros if not given
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)
            hidden = hidden.to(DEVICE)

        outs = []

        for i in range(t):
            # Concat previous hidden layer with input
            inp = torch.cat([x[:, i, :], hidden], dim=1)

            # First layer output is our new hidden
            hidden = self.lin1(inp)
            # Output layer
            out = self.lin2(hidden)

            outs.append(out[:, None, :])

        return torch.cat(outs, dim=1), hidden


class RNN(nn.Module):
    def __init__(
        self, n_embeddings: int, embedding_size: int, hidden: int, num_classes: int
    ) -> None:
        super().__init__()

        self.elman = nn.Sequential(
            nn.Embedding(n_embeddings, embedding_size), Elman(embedding_size, hidden)
        )

        self.linear = nn.Sequential(
            nn.ReLU(),
            GlobalMaxPool(1),
            nn.Linear(hidden, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        out, _ = self.elman(x)
        out = self.linear(out)

        return out
