import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class GlobalMaxPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class IMDBDataset(Dataset):
    def __init__(self, x, y, padding_value=0) -> None:
        super().__init__()

        x = [torch.tensor(xi) for xi in x]
        self.x = pad_sequence(x, batch_first=True, padding_value=padding_value)

        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]